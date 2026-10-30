//! Portable native CPU inference for decoder bundles with the resolved tensor layout.
//!
//! This backend deliberately uses only Rust, mmap, rayon and safetensors. It is
//! the universal correctness path; architecture-specific SIMD can replace the
//! matvec kernel without changing the adapter contract.

use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use half::bf16;
use kvslim_adapter_api::{AdapterCapabilities, ModelAdapter};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};

use crate::{
    gpu_canonical_lod::{CanonicalGpuLod, CanonicalGpuLodConfig},
    model_bundle::ModelBundle,
    model_config::ModelSpec,
    portable::AttentionGeometry,
};

struct WeightShard {
    path: PathBuf,
    map: Mmap,
}

impl WeightShard {
    fn open(path: PathBuf) -> Result<Self> {
        let file =
            File::open(&path).with_context(|| format!("no se pudo abrir {}", path.display()))?;
        let map = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("no se pudo mapear {}", path.display()))?;
        SafeTensors::deserialize(&map)
            .with_context(|| format!("safetensors inválido: {}", path.display()))?;
        Ok(Self { path, map })
    }

    fn names(&self) -> Result<Vec<String>> {
        Ok(SafeTensors::deserialize(&self.map)?
            .names()
            .into_iter()
            .cloned()
            .collect())
    }
}

struct NativeWeightArchive {
    shards: Vec<WeightShard>,
    owners: HashMap<String, usize>,
}

impl NativeWeightArchive {
    fn open(model_dir: &Path, shard_names: &[String]) -> Result<Self> {
        let shards = shard_names
            .iter()
            .map(|name| WeightShard::open(model_dir.join(name)))
            .collect::<Result<Vec<_>>>()?;
        let mut owners = HashMap::new();
        for (index, shard) in shards.iter().enumerate() {
            for name in shard.names()? {
                ensure!(
                    owners.insert(name.clone(), index).is_none(),
                    "tensor duplicado entre shards: {name}"
                );
            }
        }
        Ok(Self { shards, owners })
    }

    fn with_bf16<T>(
        &self,
        name: &str,
        expected: &[usize],
        f: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        let index = self
            .owners
            .get(name)
            .copied()
            .with_context(|| format!("peso ausente: {name}"))?;
        let shard = &self.shards[index];
        let archive = SafeTensors::deserialize(&shard.map)
            .with_context(|| format!("safetensors inválido: {}", shard.path.display()))?;
        let tensor = archive
            .tensor(name)
            .with_context(|| format!("peso ausente: {name}"))?;
        ensure!(tensor.dtype() == Dtype::BF16, "{name} no es BF16");
        ensure!(
            tensor.shape() == expected,
            "geometría incorrecta para {name}"
        );
        f(tensor.data())
    }

    fn vector(&self, name: &str, size: usize) -> Result<Vec<f32>> {
        self.with_bf16(name, &[size], |bytes| {
            Ok(bytes
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
                .collect())
        })
    }

    fn embedding_row(&self, name: &str, rows: usize, cols: usize, row: usize) -> Result<Vec<f32>> {
        ensure!(row < rows, "token id fuera del vocabulario");
        self.with_bf16(name, &[rows, cols], |bytes| {
            let start = row * cols * 2;
            Ok(bytes[start..start + cols * 2]
                .chunks_exact(2)
                .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
                .collect())
        })
    }

    fn matvec(&self, name: &str, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>> {
        ensure!(x.len() == cols, "entrada incorrecta para {name}");
        self.with_bf16(name, &[rows, cols], |bytes| {
            let row_bytes = cols * 2;
            Ok(bytes
                .par_chunks_exact(row_bytes)
                .map(|row| {
                    let sum = row
                        .chunks_exact(2)
                        .zip(x)
                        .map(|(w, &v)| {
                            bf16::from_bits(u16::from_le_bytes([w[0], w[1]])).to_f32() * v
                        })
                        .sum::<f32>();
                    round_bf16(sum)
                })
                .collect())
        })
    }
}

trait NativeCausalRuntime: Send {
    fn position(&self) -> usize;
    fn forward_token(&mut self, token: u32, capture_logits: bool) -> Result<u32>;
    fn last_logits(&self) -> Option<Vec<f32>>;
    fn session_tokens(&self) -> &[u32];
    fn kv_report(&self) -> Value;
    fn reset(&mut self) -> Result<()>;
}

struct NativeCpuDecoder {
    spec: ModelSpec,
    weights: NativeWeightArchive,
    kv: CanonicalGpuLod,
    kv_config: CanonicalGpuLodConfig,
    hot_keys: Vec<Vec<u8>>,
    hot_values: Vec<Vec<u8>>,
    hot_budget_bytes: usize,
    hot_active: bool,
    hot_dense_calls: u64,
    dimensional_calls: u64,
    tokens: Vec<u32>,
    last_logits: Option<Vec<f32>>,
}

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn round_vec(values: &mut [f32]) {
    values.iter_mut().for_each(|x| *x = round_bf16(*x));
}

fn add_residual(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter()
        .zip(rhs)
        .map(|(&a, &b)| round_bf16(a + b))
        .collect()
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| {
            let normalized = round_bf16(v * scale);
            round_bf16(normalized * w)
        })
        .collect()
}

fn apply_rope(values: &mut [f32], heads: usize, dim: usize, position: usize, theta: f32) {
    let half = dim / 2;
    for head in 0..heads {
        let base = head * dim;
        for i in 0..half {
            let angle = position as f32 / theta.powf((2 * i) as f32 / dim as f32);
            let (sin, cos) = angle.sin_cos();
            let sin = round_bf16(sin);
            let cos = round_bf16(cos);
            let a = values[base + i];
            let b = values[base + i + half];
            values[base + i] = round_bf16(round_bf16(a * cos) - round_bf16(b * sin));
            values[base + i + half] = round_bf16(round_bf16(b * cos) + round_bf16(a * sin));
        }
    }
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn auto_hot_kv_budget() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("MemAvailable:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<usize>().ok())
        })
        .and_then(|kib| kib.checked_mul(1024))
        .map(|bytes| bytes / 16)
        .unwrap_or(0)
}

fn dense_attention_bf16(
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query: &[f32],
    keys: &[u8],
    values: &[u8],
    tokens: usize,
) -> Result<Vec<f32>> {
    let token_bytes = kv_heads * head_dim * 2;
    ensure!(
        query.len() == query_heads * head_dim
            && keys.len() == tokens * token_bytes
            && values.len() == tokens * token_bytes,
        "geometría KV caliente incompatible"
    );
    let head_bytes = head_dim * 2;
    let group = query_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0_f32; query_heads * head_dim];
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(qh, out)| {
            let kh = qh / group;
            let q = &query[qh * head_dim..(qh + 1) * head_dim];
            let mut scores = (0..tokens)
                .map(|token| {
                    let start = token * token_bytes + kh * head_bytes;
                    q.iter()
                        .zip(keys[start..start + head_bytes].chunks_exact(2))
                        .map(|(&a, bits)| {
                            a * bf16::from_bits(u16::from_le_bytes([bits[0], bits[1]])).to_f32()
                        })
                        .sum::<f32>()
                        * scale
                })
                .collect::<Vec<_>>();
            softmax(&mut scores);
            for (token, probability) in scores.into_iter().enumerate() {
                let start = token * token_bytes + kh * head_bytes;
                for (dimension, bits) in values[start..start + head_bytes]
                    .chunks_exact(2)
                    .enumerate()
                {
                    out[dimension] += probability
                        * bf16::from_bits(u16::from_le_bytes([bits[0], bits[1]])).to_f32();
                }
            }
            round_vec(out);
        });
    Ok(output)
}

impl NativeCpuDecoder {
    fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let bundle = ModelBundle::load(model_dir)?;
        bundle.validate_decoder_layout()?;
        let spec = bundle.spec;
        ensure!(
            spec.declared_dtype.as_deref() == Some("bfloat16"),
            "la CPU nativa requiere pesos BF16"
        );
        let weights = NativeWeightArchive::open(model_dir, &spec.weight_shards)?;
        let epsilon = config
            .get("kvslim_epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let stage_tokens = config
            .get("kvslim_stage_tokens")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("kvslim_stage_tokens fuera de rango")?
            .unwrap_or(64);
        let geometry = AttentionGeometry::new(
            spec.attention.query_heads,
            spec.attention.kv_heads,
            spec.attention.head_dim,
        )?;
        let kv_config = CanonicalGpuLodConfig {
            epsilon,
            stage_tokens,
            cold_directory: None,
        };
        let kv = CanonicalGpuLod::new(geometry, spec.num_hidden_layers, kv_config.clone())?;
        let hot_budget_bytes = config
            .get("native_hot_kv_bytes")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .context("native_hot_kv_bytes fuera de rango")?
            .unwrap_or_else(auto_hot_kv_budget);
        let layers = spec.num_hidden_layers;
        Ok(Self {
            spec,
            weights,
            kv,
            kv_config,
            hot_keys: (0..layers).map(|_| Vec::new()).collect(),
            hot_values: (0..layers).map(|_| Vec::new()).collect(),
            hot_budget_bytes,
            hot_active: hot_budget_bytes > 0,
            hot_dense_calls: 0,
            dimensional_calls: 0,
            tokens: Vec::new(),
            last_logits: None,
        })
    }

    fn attend(&mut self, layer: usize, query: &[f32]) -> Result<Vec<f32>> {
        let g = self.spec.attention;
        let total = self.kv.layer_tokens(layer);
        ensure!(total > 0, "atención nativa sin KV");
        if self.hot_active {
            let token_bytes = g.kv_heads * g.head_dim * 2;
            ensure!(
                self.hot_keys[layer].len() == total * token_bytes
                    && self.hot_values[layer].len() == total * token_bytes,
                "caché KV caliente incompleta"
            );
            self.kv.observe_full_hot_attention(layer, total)?;
            self.hot_dense_calls = self.hot_dense_calls.saturating_add(1);
            return dense_attention_bf16(
                g.query_heads,
                g.kv_heads,
                g.head_dim,
                query,
                &self.hot_keys[layer],
                &self.hot_values[layer],
                total,
            );
        }
        self.dimensional_calls = self.dimensional_calls.saturating_add(1);
        let query_bytes = query
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let regional = self.kv.plan_micro_regions(layer, &query_bytes)?;
        let scored_positions = regional
            .as_ref()
            .map(|selection| selection.positions.clone())
            .unwrap_or_else(|| (0..total).collect());
        let keys = self.kv.read_keys(layer, &scored_positions)?;
        let head_bytes = g.head_dim * 2;
        let mut scored = vec![vec![0.0f32; scored_positions.len()]; g.query_heads];
        for qh in 0..g.query_heads {
            let kh = qh / (g.query_heads / g.kv_heads);
            let q = &query[qh * g.head_dim..(qh + 1) * g.head_dim];
            for (local, score) in scored[qh].iter_mut().enumerate() {
                let start = (local * g.kv_heads + kh) * head_bytes;
                let bytes = &keys[start..start + head_bytes];
                let dot = q
                    .iter()
                    .zip(bytes.chunks_exact(2))
                    .map(|(&a, value)| {
                        a * bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32()
                    })
                    .sum::<f32>();
                *score = dot / (g.head_dim as f32).sqrt();
            }
        }
        let (positions, mut selected_scores) = if regional.is_some() {
            (scored_positions, scored)
        } else {
            let flat_scores = scored.iter().flatten().copied().collect::<Vec<_>>();
            let selection = self.kv.select(layer, &query_bytes, &flat_scores)?;
            let selected = scored
                .iter()
                .map(|head| {
                    selection
                        .positions
                        .iter()
                        .map(|&token| head[token])
                        .collect()
                })
                .collect();
            (selection.positions, selected)
        };
        let values = self.kv.read_values(layer, &positions)?;
        let mut output = vec![0.0f32; g.query_heads * g.head_dim];
        for qh in 0..g.query_heads {
            let kh = qh / (g.query_heads / g.kv_heads);
            softmax(&mut selected_scores[qh]);
            for (selected, probability) in selected_scores[qh].iter().copied().enumerate() {
                let start = (selected * g.kv_heads + kh) * head_bytes;
                for (d, value) in values[start..start + head_bytes]
                    .chunks_exact(2)
                    .enumerate()
                {
                    output[qh * g.head_dim + d] += probability
                        * bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32();
                }
            }
        }
        round_vec(&mut output);
        Ok(output)
    }

    fn append_kv_page(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<()> {
        let encode = |values: &[f32]| {
            values
                .iter()
                .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        };
        let key = encode(key);
        let value = encode(value);
        if self.hot_active {
            let bytes_per_token_all_layers = (key.len() + value.len())
                .checked_mul(self.spec.num_hidden_layers)
                .context("presupuesto KV caliente desbordado")?;
            let projected = (self.tokens.len() + 1)
                .checked_mul(bytes_per_token_all_layers)
                .context("tamaño KV caliente desbordado")?;
            if projected <= self.hot_budget_bytes {
                self.hot_keys[layer].extend_from_slice(&key);
                self.hot_values[layer].extend_from_slice(&value);
            } else {
                self.hot_active = false;
                self.hot_keys.iter_mut().for_each(Vec::clear);
                self.hot_values.iter_mut().for_each(Vec::clear);
            }
        }
        self.kv.append(layer, self.tokens.len(), &key, &value)
    }

    fn forward_token(&mut self, token: u32, capture_logits: bool) -> Result<u32> {
        let hidden = self.spec.hidden_size;
        let intermediate = self
            .spec
            .intermediate_size
            .context("falta intermediate_size")?;
        let vocab = self.spec.vocab_size.context("falta vocab_size")?;
        let g = self.spec.attention;
        let position = self.tokens.len();
        let eps = self.spec.rms_norm_eps.unwrap_or(1e-5) as f32;
        let theta = self.spec.rope_theta.unwrap_or(10_000.0) as f32;

        let mut state = self.weights.embedding_row(
            "model.embed_tokens.weight",
            vocab,
            hidden,
            token as usize,
        )?;
        for layer in 0..self.spec.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let norm = self
                .weights
                .vector(&format!("{prefix}.input_layernorm.weight"), hidden)?;
            let normalized = rms_norm(&state, &norm, eps);
            let mut query = self.weights.matvec(
                &format!("{prefix}.self_attn.q_proj.weight"),
                g.query_heads * g.head_dim,
                hidden,
                &normalized,
            )?;
            let mut key = self.weights.matvec(
                &format!("{prefix}.self_attn.k_proj.weight"),
                g.kv_heads * g.head_dim,
                hidden,
                &normalized,
            )?;
            let mut value = self.weights.matvec(
                &format!("{prefix}.self_attn.v_proj.weight"),
                g.kv_heads * g.head_dim,
                hidden,
                &normalized,
            )?;
            round_vec(&mut query);
            round_vec(&mut key);
            round_vec(&mut value);
            apply_rope(&mut query, g.query_heads, g.head_dim, position, theta);
            apply_rope(&mut key, g.kv_heads, g.head_dim, position, theta);
            self.append_kv_page(layer, &key, &value)?;
            let attention = self.attend(layer, &query)?;
            let projected = self.weights.matvec(
                &format!("{prefix}.self_attn.o_proj.weight"),
                hidden,
                g.query_heads * g.head_dim,
                &attention,
            )?;
            state = add_residual(&state, &projected);

            let post_norm = self
                .weights
                .vector(&format!("{prefix}.post_attention_layernorm.weight"), hidden)?;
            let normalized = rms_norm(&state, &post_norm, eps);
            let gate = self.weights.matvec(
                &format!("{prefix}.mlp.gate_proj.weight"),
                intermediate,
                hidden,
                &normalized,
            )?;
            let up = self.weights.matvec(
                &format!("{prefix}.mlp.up_proj.weight"),
                intermediate,
                hidden,
                &normalized,
            )?;
            let gated: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| {
                    let silu = round_bf16(g / (1.0 + (-g).exp()));
                    round_bf16(silu * u)
                })
                .collect();
            let down = self.weights.matvec(
                &format!("{prefix}.mlp.down_proj.weight"),
                hidden,
                intermediate,
                &gated,
            )?;
            state = add_residual(&state, &down);
        }

        self.tokens.push(token);
        let norm = self.weights.vector("model.norm.weight", hidden)?;
        let state = rms_norm(&state, &norm, eps);
        let logits = self
            .weights
            .matvec("lm_head.weight", vocab, hidden, &state)?;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as u32)
            .context("vocabulario vacío")?;
        self.last_logits = capture_logits.then_some(logits);
        Ok(next)
    }
}

impl NativeCausalRuntime for NativeCpuDecoder {
    fn position(&self) -> usize {
        self.tokens.len()
    }

    fn forward_token(&mut self, token: u32, capture_logits: bool) -> Result<u32> {
        NativeCpuDecoder::forward_token(self, token, capture_logits)
    }

    fn last_logits(&self) -> Option<Vec<f32>> {
        self.last_logits.clone()
    }

    fn session_tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn kv_report(&self) -> Value {
        let stats = self.kv.storage_metrics();
        let lod = &self.kv.stats;
        json!({
            "authority": "kvslim-cold-store",
            "format": "exact-bf16",
            "pages": stats.pages,
            "committed_bytes": stats.committed_bytes,
            "reads": stats.reads,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "hot_bytes": stats.hot_bytes,
            "hot_capacity": stats.hot_capacity,
            "native_hot_budget_bytes": self.hot_budget_bytes,
            "native_hot_resident_bytes": self.hot_keys.iter().map(Vec::len).sum::<usize>()
                + self.hot_values.iter().map(Vec::len).sum::<usize>(),
            "native_hot_active": self.hot_active,
            "native_hot_dense_calls": self.hot_dense_calls,
            "native_dimensional_calls": self.dimensional_calls,
            "route_selector": "measured-residency-first",
            "epsilon": self.kv.config().epsilon,
            "axis_tokens": self.kv.axis().token_len(),
            "axis_extents": self.kv.axis().regions().len(),
            "micro_metadata_bytes": self.kv.axis().micro_metadata_bytes(),
            "four_bit_index_bytes": self.kv.axis().fingerprint_payload_bytes(),
            "certificates_met": lod.certificates_met,
            "dense_fallbacks": lod.exact_fallbacks + lod.micro_dense_fallback_queries,
            "selected_v_tokens": lod.exact_v_tokens_read_from_cold,
            "omitted_v_tokens": lod.exact_v_tokens_skipped,
            "partial_key_reads": stats.partial_key_reads,
            "partial_value_reads": stats.partial_value_reads,
            "partial_key_bytes_avoided": stats.partial_key_bytes_avoided,
            "partial_value_bytes_avoided": stats.partial_value_bytes_avoided,
        })
    }

    fn reset(&mut self) -> Result<()> {
        let g = self.spec.attention;
        self.kv = CanonicalGpuLod::new(
            AttentionGeometry::new(g.query_heads, g.kv_heads, g.head_dim)?,
            self.spec.num_hidden_layers,
            self.kv_config.clone(),
        )?;
        self.hot_keys.iter_mut().for_each(Vec::clear);
        self.hot_values.iter_mut().for_each(Vec::clear);
        self.hot_active = self.hot_budget_bytes > 0;
        self.hot_dense_calls = 0;
        self.dimensional_calls = 0;
        self.tokens.clear();
        self.last_logits = None;
        Ok(())
    }
}

fn build_runtime(
    model_dir: &Path,
    _spec: &ModelSpec,
    config: &Value,
) -> Result<Box<dyn NativeCausalRuntime>> {
    Ok(Box::new(NativeCpuDecoder::load(model_dir, config)?))
}

pub struct NativeCpuModelAdapter {
    capabilities: AdapterCapabilities,
    model_dir: Option<PathBuf>,
    runtime: Option<Box<dyn NativeCausalRuntime>>,
}

impl Default for NativeCpuModelAdapter {
    fn default() -> Self {
        Self {
            capabilities: AdapterCapabilities {
                adapter_id: "native-cpu-runtime".into(),
                backend: "cpu".into(),
                supports_prefill: true,
                supports_decode: true,
                supports_kv_restore: true,
                supports_streaming: true,
            },
            model_dir: None,
            runtime: None,
        }
    }
}

impl ModelAdapter for NativeCpuModelAdapter {
    fn capabilities(&self) -> &AdapterCapabilities {
        &self.capabilities
    }

    fn inspect(&self, model_dir: &Path, _config: &Value) -> Result<()> {
        let bundle = ModelBundle::load(model_dir)?;
        bundle.validate_decoder_layout()?;
        ensure!(
            bundle.spec.declared_dtype.as_deref() == Some("bfloat16"),
            "CPU nativa requiere BF16"
        );
        Ok(())
    }

    fn load(&mut self, model_dir: &Path, config: &Value) -> Result<()> {
        self.inspect(model_dir, config)?;
        let spec = ModelSpec::inspect(model_dir)?;
        self.runtime = Some(build_runtime(model_dir, &spec, config)?);
        self.model_dir = Some(model_dir.to_owned());
        Ok(())
    }

    fn prefill(&mut self, input: &[u32]) -> Result<Value> {
        ensure!(!input.is_empty(), "prefill vacío");
        let runtime = self.runtime.as_mut().context("CPU nativa no cargada")?;
        for (index, &token) in input.iter().enumerate() {
            runtime.forward_token(token, index + 1 == input.len())?;
        }
        Ok(json!({
            "tokens": input.len(),
            "position": runtime.position(),
            "kv": runtime.kv_report(),
        }))
    }

    fn decode(&mut self, token_id: u32) -> Result<u32> {
        self.runtime
            .as_mut()
            .context("CPU nativa no cargada")?
            .forward_token(token_id, true)
    }

    fn last_logits(&self) -> Result<Option<Vec<f32>>> {
        Ok(self.runtime.as_ref().and_then(|r| r.last_logits()))
    }

    fn snapshot_kv(&self) -> Result<Vec<u8>> {
        let runtime = self.runtime.as_ref().context("CPU nativa no cargada")?;
        serde_json::to_vec(&json!({
            "schema": "kvslim.runtime.replay-snapshot/v1",
            "model_dir": self.model_dir,
            "tokens": runtime.session_tokens(),
        }))
        .context("serializar snapshot CPU nativa")
    }

    fn restore_kv(&mut self, snapshot: &[u8]) -> Result<()> {
        let value: Value = serde_json::from_slice(snapshot)?;
        ensure!(
            value.get("schema").and_then(Value::as_str)
                == Some("kvslim.runtime.replay-snapshot/v1"),
            "snapshot no compatible"
        );
        let expected = value
            .get("model_dir")
            .and_then(Value::as_str)
            .context("snapshot sin identidad")?;
        ensure!(
            self.model_dir.as_deref() == Some(Path::new(expected)),
            "modelo del snapshot no coincide"
        );
        let tokens = value
            .get("tokens")
            .and_then(Value::as_array)
            .context("snapshot sin tokens")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .context("token inválido")
            })
            .collect::<Result<Vec<_>>>()?;
        let runtime = self.runtime.as_mut().context("CPU nativa no cargada")?;
        runtime.reset()?;
        for token in tokens {
            runtime.forward_token(token, false)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_is_finite_and_bf16_rounded() {
        let out = rms_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 1e-5);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out.iter().all(|&x| bf16::from_f32(x).to_f32() == x));
    }

    #[test]
    fn rope_at_zero_is_identity() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 4, 0, 10_000.0);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn softmax_is_normalized() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
