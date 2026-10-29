//! Batch prefill kernels and host launch plan for the original BF16
//! MiniCPM5-1B GPU engine. This module has NO CPU/LibTorch dependencies.
//!
//! Layer-major across a tile of prompt tokens, not 24 layers per token.
//! Causal attention masks keys with positions after the current token.
//! Original BF16 weight matrices, LoRA-free GEMV summation and KV are used.
//! At the end of GPU prefill, LOD mode seals exact K/V into the ORIGINAL
//! canonical Rust ColdStore in 64-token pages, without per-token GPU V reads.
//! During subsequent decode, only K stays resident on GPU; V is retrieved
//! through the existing exact canonical ColdStore with verification.
//!
//! Intentional constraints:
//! - batch kernels are optional until parity is proven;
//! - optional up to 512 tokens per GPU batch; scratch scores <=64MiB;
//! - CPU branch-and-bound remains separate/unmodified;
//! - numerical contract is GPU dense/Lod epsilon-zero parity, not LibTorch.

use super::*;
use opencl3::kernel::Kernel;

const MAX_SCRATCH_SCORES_BYTES: usize = 64 * 1024 * 1024;
const COLD_PAGE_TOKENS: usize = 64;

pub(super) struct BatchKernels {
    embed: Kernel,
    gemv: Kernel,
    gemv_reuse8: Kernel,
    gemv_reuse16: Kernel,
    gemv_reuse32: Kernel,
    gemv_register8: Kernel,
    gemv_register16: Kernel,
    gemv_virtual64: Kernel,
    gemv_virtual32: Kernel,
    fused_qkv8: Kernel,
    fused_swiglu8: Kernel,
    fusion_mask: u8, // bit0 QKV, bit1 BF16 gate/up/SwiGLU
    register_unrolled: bool,
    virtual_lanes: usize,
    weight_reuse_tokens: usize,
    norm: Kernel,
    rope: Kernel,
    qk: Kernel,
    softmax: Kernel,
    pv: Kernel,
}

// Low-level OpenCL bindings intentionally mirror the kernel argument arity.
#[allow(clippy::too_many_arguments)]
impl BatchKernels {
    pub(super) fn load(
        engine: &GpuEngine,
        reuse: usize,
        register_unrolled: bool,
        virtual_lanes: usize,
        fusion_mask: u8,
    ) -> Result<Self> {
        ensure!(
            [32, 64, 128].contains(&virtual_lanes),
            "physical OpenCL prefill lanes must be 32,64,128"
        );
        ensure!(
            virtual_lanes == 128 || (reuse == 8 && !register_unrolled),
            "experimental virtual-lane prefill requires 8 tokens and no register unroll"
        );
        ensure!(
            !register_unrolled || [8, 16].contains(&reuse),
            "register-unrolled BF16 prefill requires 8 or 16 tokens per weight tile"
        );
        ensure!(
            [1, 8, 16, 32].contains(&reuse),
            "GPU BF16 weight reuse must be 1,8,16,32"
        );
        ensure!(fusion_mask <= 3, "GPU BF16 fused prefill mask invalid");
        ensure!(
            fusion_mask == 0 || (reuse == 8 && !register_unrolled && virtual_lanes == 128),
            "BF16 fused prefill projections require original 128-lane reuse8, no register unroll"
        );
        let get = |name| {
            Kernel::create(&engine._program, name)
                .with_context(|| format!("GPU prefill batch kernel absent: {name}"))
        };
        Ok(Self {
            embed: get("llm_prefill_embed")?,
            gemv: get("llm_prefill_gemv")?,
            gemv_reuse8: get("llm_prefill_gemv_reuse8")?,
            gemv_reuse16: get("llm_prefill_gemv_reuse16")?,
            gemv_reuse32: get("llm_prefill_gemv_reuse32")?,
            gemv_register8: get("llm_prefill_gemv_reuse8_register")?,
            gemv_register16: get("llm_prefill_gemv_reuse16_register")?,
            gemv_virtual64: get("llm_prefill_gemv_reuse8_virtual64")?,
            gemv_virtual32: get("llm_prefill_gemv_reuse8_virtual32")?,
            fused_qkv8: get("llm_prefill_fused_qkv_reuse8")?,
            fused_swiglu8: get("llm_prefill_fused_swiglu_reuse8")?,
            fusion_mask,
            register_unrolled,
            virtual_lanes,
            weight_reuse_tokens: reuse,
            norm: get("llm_prefill_norm")?,
            rope: get("llm_prefill_rope")?,
            qk: get("llm_prefill_qk")?,
            softmax: get("llm_prefill_softmax")?,
            pv: get("llm_prefill_pv")?,
        })
    }

    fn embed(
        &self,
        engine: &GpuEngine,
        weight: &GpuWeight,
        ids: &Buffer<u32>,
        output: &Buffer<u8>,
        tokens: usize,
        hidden: usize,
    ) -> Result<()> {
        let n = u32::try_from(tokens)?;
        let h = u32::try_from(hidden)?;
        // SAFETY: GPU buffers sized to [tokens,hidden], and kernel guards bounds.
        unsafe {
            ExecuteKernel::new(&self.embed)
                .set_arg(&weight.data)
                .set_arg(ids)
                .set_arg(output)
                .set_arg(&n)
                .set_arg(&h)
                .set_global_work_size(tokens * hidden)
                .enqueue_nd_range(&engine.queue)
                .context("GPU batch prefill embedding failed")?;
        }
        Ok(())
    }

    pub(super) fn gemv(
        &self,
        engine: &GpuEngine,
        weight: &GpuWeight,
        input: &Buffer<u8>,
        output: &Buffer<u8>,
        in_dim: usize,
        out_dim: usize,
        tokens: usize,
    ) -> Result<()> {
        #[cfg(feature = "hip-backend")]
        if let Some(ref hip) = weight.hip {
            return engine.hip_matrix(hip, input, output, in_dim, out_dim, tokens);
        }
        let (d, o, t) = (
            u32::try_from(in_dim)?,
            u32::try_from(out_dim)?,
            u32::try_from(tokens)?,
        );
        let global = [
            out_dim * self.virtual_lanes,
            tokens.div_ceil(self.weight_reuse_tokens),
        ];
        let local = [self.virtual_lanes, 1usize];
        let kernel = match (
            self.weight_reuse_tokens,
            self.register_unrolled,
            self.virtual_lanes,
        ) {
            (8, false, 64) => &self.gemv_virtual64,
            (8, false, 32) => &self.gemv_virtual32,
            (8, true, 128) => &self.gemv_register8,
            (16, true, 128) => &self.gemv_register16,
            (1, false, 128) => &self.gemv,
            (8, false, 128) => &self.gemv_reuse8,
            (16, false, 128) => &self.gemv_reuse16,
            (32, false, 128) => &self.gemv_reuse32,
            _ => unreachable!("prefill BF16 kernel geometry validated when loaded"),
        };
        // SAFETY: Row-major BF16 weight and input dimensions verified by
        // the original loader; each output (row,token) uses the same 128
        // lane reduction as the original single-token BF16 GEMV kernel.
        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&weight.data)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&d)
                .set_arg(&o)
                .set_arg(&t)
                .set_global_work_sizes(&global)
                .set_local_work_sizes(&local)
                .enqueue_nd_range(&engine.queue)
                .with_context(|| format!("GPU batch GEMV [{out_dim},{in_dim}] x {tokens}"))?;
        }
        Ok(())
    }

    /// Single physical AMD OpenCL launch for exact BF16 Q, K and V
    /// weight projections for all eight prompt tokens in a tile.
    /// All GPU accumulators and final BF16 rounding preserve the
    /// original three independent 128-lane GEMV reduction trees.
    pub(super) fn project_qkv8(
        &self,
        engine: &GpuEngine,
        weights: (&GpuWeight, &GpuWeight, &GpuWeight),
        buffers: (&Buffer<u8>, &Buffer<u8>, &Buffer<u8>, &Buffer<u8>),
        shape: (usize, usize, usize, usize),
    ) -> Result<()> {
        let (wq, wk, wv) = weights;
        let (input, q, k, v) = buffers;
        let (hidden, qrows, kvrows, tokens) = shape;
        ensure!(
            qrows >= kvrows && tokens > 0,
            "GPU fused prefill QKV has invalid original BF16 shape"
        );
        #[cfg(feature = "hip-backend")]
        ensure!(
            wq.hip.is_none() && wk.hip.is_none() && wv.hip.is_none(),
            "cannot use OpenCL projection fusion with HIP-owned BF16 weights"
        );
        let (h, q_count, kv_count, t) = (
            u32::try_from(hidden)?,
            u32::try_from(qrows)?,
            u32::try_from(kvrows)?,
            u32::try_from(tokens)?,
        );
        let globals = [qrows * LOCAL_WORK_SIZE, tokens.div_ceil(8)];
        let local = [LOCAL_WORK_SIZE, 1usize];
        // SAFETY: Original safetensor shapes are validated by model
        // loader. BF16 inputs are [token,hidden], Q/K/V outputs
        // remain [token,row] as in original GEMV kernels.
        unsafe {
            ExecuteKernel::new(&self.fused_qkv8)
                .set_arg(&wq.data)
                .set_arg(&wk.data)
                .set_arg(&wv.data)
                .set_arg(input)
                .set_arg(q)
                .set_arg(k)
                .set_arg(v)
                .set_arg(&h)
                .set_arg(&q_count)
                .set_arg(&kv_count)
                .set_arg(&t)
                .set_global_work_sizes(&globals)
                .set_local_work_sizes(&local)
                .enqueue_nd_range(&engine.queue)
                .context("GPU BF16 prefill fused QKV 8-token projection failed")?;
        }
        Ok(())
    }

    /// Original BF16 gate+up GEMV, intermediate BF16 roundings and
    /// SiLU/product fused into one actual GPU launch per token tile.
    pub(super) fn project_swiglu8(
        &self,
        engine: &GpuEngine,
        weights: (&GpuWeight, &GpuWeight),
        normalized: &Buffer<u8>,
        output: &Buffer<u8>,
        shape: (usize, usize, usize),
    ) -> Result<()> {
        let (gate, up) = weights;
        let (hidden, intermediate, tokens) = shape;
        ensure!(
            tokens > 0 && hidden > 0 && intermediate > 0,
            "GPU BF16 fused prefill SwiGLU dimensions invalid"
        );
        #[cfg(feature = "hip-backend")]
        ensure!(
            gate.hip.is_none() && up.hip.is_none(),
            "cannot fuse OpenCL BF16 SwiGLU with HIP-owned matrices"
        );
        let (h, m, t) = (
            u32::try_from(hidden)?,
            u32::try_from(intermediate)?,
            u32::try_from(tokens)?,
        );
        let global = [intermediate * LOCAL_WORK_SIZE, tokens.div_ceil(8)];
        let local = [LOCAL_WORK_SIZE, 1usize];
        // SAFETY: Same original BF16 gate/up weight layout and output
        // [token,intermediate], each token independently causal.
        unsafe {
            ExecuteKernel::new(&self.fused_swiglu8)
                .set_arg(&gate.data)
                .set_arg(&up.data)
                .set_arg(normalized)
                .set_arg(output)
                .set_arg(&h)
                .set_arg(&m)
                .set_arg(&t)
                .set_global_work_sizes(&global)
                .set_local_work_sizes(&local)
                .enqueue_nd_range(&engine.queue)
                .context("GPU BF16 prefill fused SwiGLU 8-token projection failed")?;
        }
        Ok(())
    }

    fn norm(
        &self,
        engine: &GpuEngine,
        input: &Buffer<u8>,
        weight: &GpuWeight,
        output: &Buffer<u8>,
        hidden: usize,
        tokens: usize,
        eps: f32,
    ) -> Result<()> {
        let h = u32::try_from(hidden)?;
        let t = u32::try_from(tokens)?;
        // SAFETY: A 128-lane workgroup per token, same algorithm as
        // original llm_rmsnorm, BF16 output written independently.
        unsafe {
            ExecuteKernel::new(&self.norm)
                .set_arg(input)
                .set_arg(&weight.data)
                .set_arg(output)
                .set_arg(&h)
                .set_arg(&t)
                .set_arg(&eps)
                .set_global_work_size(tokens * LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&engine.queue)
                .context("GPU batch prefill RMSNorm failed")?;
        }
        Ok(())
    }

    fn rope(
        &self,
        engine: &GpuEngine,
        input: &Buffer<u8>,
        output: &Buffer<u8>,
        tokens: usize,
        heads: usize,
        dim: usize,
        prior: usize,
        rope: RopeKernelConfig,
    ) -> Result<()> {
        let (t, h, d, pos) = (
            u32::try_from(tokens)?,
            u32::try_from(heads)?,
            u32::try_from(dim)?,
            u32::try_from(prior)?,
        );
        // SAFETY: All query/token dimensions and absolute RoPE positions
        // are checked before launch; identical FP32/BF16 sequence to
        // the original single-token llm_rope kernel.
        unsafe {
            ExecuteKernel::new(&self.rope)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&t)
                .set_arg(&h)
                .set_arg(&d)
                .set_arg(&pos)
                .set_arg(&rope.theta)
                .set_arg(&rope.llama3_scaling)
                .set_arg(&rope.factor)
                .set_arg(&rope.low_freq_factor)
                .set_arg(&rope.high_freq_factor)
                .set_arg(&rope.original_max_position)
                .set_global_work_size(tokens * heads * dim)
                .enqueue_nd_range(&engine.queue)
                .context("GPU batch prefill rotary embeddings failed")?;
        }
        Ok(())
    }

    fn attention(
        &self,
        engine: &GpuEngine,
        queries: &Buffer<u8>,
        keys: &Buffer<u8>,
        values: &Buffer<u8>,
        scores: &Buffer<f32>,
        output: &Buffer<u8>,
        tokens: usize,
        prior: usize,
        geom: AttentionGeometry,
    ) -> Result<()> {
        let history = prior
            .checked_add(tokens)
            .context("GPU causal history overflow")?;
        let (b, p, n, q, k, d) = (
            u32::try_from(tokens)?,
            u32::try_from(prior)?,
            u32::try_from(history)?,
            u32::try_from(geom.query_heads)?,
            u32::try_from(geom.kv_heads)?,
            u32::try_from(geom.head_dim)?,
        );
        let items = tokens
            .checked_mul(geom.query_heads)
            .and_then(|x| x.checked_mul(history))
            .context("GPU causal scores overflow")?;
        u32::try_from(items).context("GPU causal score index exceeds u32")?;
        // SAFETY: GPU K/V have been queued to device before QK; ordered
        // queue enforces QK -> softmax -> PV. No future keys can be read.
        unsafe {
            ExecuteKernel::new(&self.qk)
                .set_arg(queries)
                .set_arg(keys)
                .set_arg(scores)
                .set_arg(&b)
                .set_arg(&p)
                .set_arg(&n)
                .set_arg(&q)
                .set_arg(&k)
                .set_arg(&d)
                .set_global_work_size(items)
                .enqueue_nd_range(&engine.queue)
                .context("GPU batch prefill causal QK failed")?;
            ExecuteKernel::new(&self.softmax)
                .set_arg(scores)
                .set_arg(&b)
                .set_arg(&p)
                .set_arg(&n)
                .set_arg(&q)
                .set_global_work_size(tokens * geom.query_heads)
                .enqueue_nd_range(&engine.queue)
                .context("GPU batch prefill causal softmax failed")?;
            ExecuteKernel::new(&self.pv)
                .set_arg(scores)
                .set_arg(values)
                .set_arg(output)
                .set_arg(&b)
                .set_arg(&p)
                .set_arg(&n)
                .set_arg(&q)
                .set_arg(&k)
                .set_arg(&d)
                .set_global_work_size(tokens * geom.query_elements())
                .enqueue_nd_range(&engine.queue)
                .context("GPU batch prefill causal PV failed")?;
        }
        Ok(())
    }
}

// All batch activations persist for one 1..512-token tile. They are
// temporary and dropped after prefill; they are NOT permanent GPU KV.
// GPU K/V histories remain in the original Workspace.kv buffers.
struct BatchBuffers {
    ids: Buffer<u32>,
    hidden0: Buffer<u8>,
    hidden1: Buffer<u8>,
    norm1: Buffer<u8>,
    norm2: Buffer<u8>,
    query: Buffer<u8>,
    query_rope: Buffer<u8>,
    key: Buffer<u8>,
    key_rope: Buffer<u8>,
    value: Buffer<u8>,
    attention: Buffer<u8>,
    attn_o: Buffer<u8>,
    gate: Buffer<u8>,
    up: Buffer<u8>,
    activation: Buffer<u8>,
    down: Buffer<u8>,
    scores: Buffer<f32>,
}

impl BatchBuffers {
    fn new(
        engine: &GpuEngine,
        spec: &ModelSpec,
        batch_limit: usize,
        prompt_len: usize,
    ) -> Result<Self> {
        let hidden = spec.hidden_size * 2;
        let query = spec.attention.query_elements() * 2;
        let kv = spec.attention.kv_elements_per_token() * 2;
        let middle = spec.intermediate_size.context("model has no MLP width")? * 2;
        let alloc = |size: usize| {
            engine.alloc_u8(
                size.checked_mul(batch_limit)
                    .context("GPU prefill activation count overflow")?,
            )
        };
        let score_items = batch_limit
            .checked_mul(spec.attention.query_heads)
            .and_then(|v| v.checked_mul(prompt_len))
            .context("GPU prefill scratch score length overflow")?;
        ensure!(
            score_items
                .checked_mul(4)
                .is_some_and(|x| x <= MAX_SCRATCH_SCORES_BYTES),
            "GPU prefill scratch exceeds 64MiB budget"
        );
        Ok(Self {
            ids: engine.alloc_indices(batch_limit)?,
            hidden0: alloc(hidden)?,
            hidden1: alloc(hidden)?,
            norm1: alloc(hidden)?,
            norm2: alloc(hidden)?,
            query: alloc(query)?,
            query_rope: alloc(query)?,
            key: alloc(kv)?,
            key_rope: alloc(kv)?,
            value: alloc(kv)?,
            attention: alloc(query)?,
            attn_o: alloc(hidden)?,
            gate: alloc(middle)?,
            up: alloc(middle)?,
            activation: alloc(middle)?,
            down: alloc(hidden)?,
            scores: engine.alloc_scores(score_items)?,
        })
    }
}

impl OpenClLlm {
    /// Processes the prompt layer-by-layer in token tiles. This replaces
    /// repeated ColdStore reads during prefill with temporary BF16 V in
    /// GPU memory, and only seals exact ColdStore pages after completion.
    ///
    /// The layout and arithmetic follow the single-token BF16 GPU route;
    /// any divergence is caught by physical-GPU parity tests. If the batch
    /// path fails, the runtime is marked poisoned by generate_cancellable.
    pub(super) fn prefill_batched(
        &mut self,
        prompt_ids: &[u32],
        restored_prefix_tokens: usize,
        requested_batch: usize,
        capture_logits: bool,
        cancel: &AtomicBool,
    ) -> Result<GpuStep> {
        ensure!(!self.poisoned, "GPU model is poisoned");
        ensure!(
            self.position == restored_prefix_tokens,
            "GPU batch prefill prefix offset must match restored exact BF16 KV"
        );
        ensure!(
            restored_prefix_tokens < prompt_ids.len(),
            "GPU batch prefill suffix must contain at least one new token"
        );
        if restored_prefix_tokens > 0 {
            ensure!(
                self.lod.as_ref().is_some_and(|x| x.config().epsilon == 0.0),
                "prefill prefix rehydration is exact BF16 canonical LOD only"
            );
            ensure!(
                self.hot_values
                    .as_ref()
                    .is_some_and(|h| h.valid_prefix_tokens == restored_prefix_tokens
                        && h.capacity_tokens >= prompt_ids.len()),
                "GPU prefill prefix requires complete verified original hot BF16 V"
            );
        }
        ensure!(!prompt_ids.is_empty(), "GPU batch prefill empty prompt");
        ensure!(
            prompt_ids.len() < self.max_tokens,
            "GPU batch prefill exceeds configured KV capacity"
        );
        let vocab = self.spec.vocab_size.context("missing model vocabulary")?;
        ensure!(
            prompt_ids.iter().all(|&id| (id as usize) < vocab),
            "GPU batch prefill contains invalid tokenizer IDs"
        );
        let start = Instant::now();
        let geom = self.spec.attention;
        let per_score_token = geom
            .query_heads
            .checked_mul(prompt_ids.len())
            .context("GPU batch prefill index overflow")?;
        let allowed_by_memory = MAX_SCRATCH_SCORES_BYTES / (4 * per_score_token);
        ensure!(
            allowed_by_memory > 0,
            "GPU prefill context too long for 64MiB scratch score budget"
        );
        let batch_limit = requested_batch
            .min(prompt_ids.len() - restored_prefix_tokens)
            .min(allowed_by_memory);
        ensure!(
            batch_limit > 0 && batch_limit <= 512,
            "GPU batch size outside tested <=512 resource cap"
        );
        let engine = &self.engine;
        let kernels = BatchKernels::load(
            engine,
            self.prefill_weight_reuse_tokens,
            self.prefill_register_unrolled,
            self.prefill_virtual_lanes,
            self.prefill_fusion_mask,
        )?;
        let mut batch = BatchBuffers::new(engine, &self.spec, batch_limit, prompt_ids.len())?;
        let kv_bytes = geom.kv_elements_per_token() * 2;
        let hidden_bytes = self.spec.hidden_size * 2;
        let mut temp_values = if self.lod.is_some() {
            // Short-lived dense V cache for prefill only. It does not remain
            // resident for generation/decode.
            let bytes = prompt_ids
                .len()
                .checked_mul(kv_bytes)
                .context("GPU temporary prefill V overflow")?;
            let mut values = Vec::with_capacity(self.spec.num_hidden_layers);
            for _ in 0..self.spec.num_hidden_layers {
                values.push(engine.alloc_u8(bytes)?);
            }
            self.last_prefill_temporary_v_bytes = bytes * self.spec.num_hidden_layers;
            Some(values)
        } else {
            self.last_prefill_temporary_v_bytes = 0;
            None
        };
        if restored_prefix_tokens > 0 {
            let prior_bytes = restored_prefix_tokens
                .checked_mul(kv_bytes)
                .context("BF16 GPU prior prefill cache overflow")?;
            let hot = self
                .hot_values
                .as_ref()
                .context("no hot BF16 V for verified prefix replay")?;
            let temporary = temp_values
                .as_mut()
                .context("no causal batch value cache for verified prefix replay")?;
            for (layer, source) in hot.values.iter().enumerate() {
                engine
                    .copy(source, &mut temporary[layer], prior_bytes, 0)
                    .with_context(|| format!("copy exact BF16 GPU prefix V layer {layer}"))?;
            }
            engine
                .queue
                .finish()
                .context("GPU restored BF16 prefix V staging barrier")?;
        }
        let eps = self.spec.rms_norm_eps.context("missing RMSNorm epsilon")? as f32;
        let rope = RopeKernelConfig::from_spec(&self.spec)?;
        let hidden = self.spec.hidden_size;
        let middle = self
            .spec
            .intermediate_size
            .context("missing MLP intermediate_size")?;
        let mut last_chunk_count = 0usize;

        for (chunk_no, ids) in prompt_ids[restored_prefix_tokens..]
            .chunks(batch_limit)
            .enumerate()
        {
            ensure!(
                !cancel.load(Ordering::Relaxed),
                "GPU batch prefill cancelled"
            );
            let prior = restored_prefix_tokens + chunk_no * batch_limit;
            let count = ids.len();
            last_chunk_count = count;
            // SAFETY: All input IDs are valid and blocking OpenCL copy holds
            // the host slice until it has been transferred.
            unsafe {
                engine
                    .queue
                    .enqueue_write_buffer(&mut batch.ids, CL_BLOCKING, 0, ids, &[])
                    .context("GPU upload prompt token IDs for batch")?;
            }
            kernels.embed(
                engine,
                &self.embedding,
                &batch.ids,
                &batch.hidden0,
                count,
                hidden,
            )?;
            for layer_index in 0..self.spec.num_hidden_layers {
                ensure!(
                    !cancel.load(Ordering::Relaxed),
                    "GPU batch prefill cancelled at layer boundary"
                );
                if self.mode == WeightMode::Resident {
                    let layer = self
                        .layers
                        .get(layer_index)
                        .context("missing resident BF16 GPU layer")?;
                    let cache = &mut self.workspace.kv[layer_index];
                    let value_cache = if let Some(temp) = temp_values.as_mut() {
                        &mut temp[layer_index]
                    } else {
                        cache.value.as_mut().context("dense GPU KV V missing")?
                    };
                    run_batch_layer(
                        engine,
                        &kernels,
                        &self.spec,
                        layer,
                        &batch,
                        &mut cache.key,
                        value_cache,
                        prior,
                        count,
                        hidden,
                        middle,
                        eps,
                        rope,
                    )?;
                } else {
                    let checkpoint = self
                        .checkpoint
                        .as_ref()
                        .context("missing streamed BF16 checkpoint")?;
                    let layer = GpuLayer::load(engine, checkpoint, layer_index, &self.spec)?;
                    let cache = &mut self.workspace.kv[layer_index];
                    let value_cache = if let Some(temp) = temp_values.as_mut() {
                        &mut temp[layer_index]
                    } else {
                        cache.value.as_mut().context("dense GPU V missing")?
                    };
                    run_batch_layer(
                        engine,
                        &kernels,
                        &self.spec,
                        &layer,
                        &batch,
                        &mut cache.key,
                        value_cache,
                        prior,
                        count,
                        hidden,
                        middle,
                        eps,
                        rope,
                    )?;
                    // No GPU kernel may reference streamed weights after
                    // buffers are dropped at the end of this iteration.
                    engine
                        .queue
                        .finish()
                        .context("GPU streamed batch weight barrier")?;
                }
            }
            self.last_prefill_batches += 1;
        }
        engine
            .queue
            .finish()
            .context("complete causal GPU batch prefill")?;
        // An optional RAM-only, exact-ID prefill replay takes ownership
        // of the SAME BF16 K/V bytes already read for the original
        // ColdStore. No additional GPU -> CPU tensor read is introduced.
        let mut replay_kv = if self.gpu_prompt_replay_max_tokens > 0
            && prompt_ids.len() <= self.gpu_prompt_replay_max_tokens
        {
            Some(Vec::with_capacity(self.spec.num_hidden_layers))
        } else {
            None
        };

        // BF16 ColdStore authority is established BEFORE the first decode
        // step. Each layer is read back only once, and the exact original
        // ColdStore page format is reused, with up to 64 tokens per page.
        if let Some(ref mut lod) = self.lod {
            let seal_started = Instant::now();
            let temporary = temp_values
                .as_ref()
                .context("temporary LOD values missing")?;
            let total_bytes = prompt_ids.len() * kv_bytes;
            for (i, value) in temporary.iter().enumerate() {
                let key_raw = engine.read_bf16_bytes(&self.workspace.kv[i].key, total_bytes)?;
                let value_raw = engine.read_bf16_bytes(value, total_bytes)?;
                let offset = restored_prefix_tokens
                    .checked_mul(kv_bytes)
                    .context("BF16 replay ColdStore slice offset overflow")?;
                lod.append_batch(
                    i,
                    restored_prefix_tokens,
                    &key_raw[offset..],
                    &value_raw[offset..],
                    COLD_PAGE_TOKENS,
                )?;
                if let Some(ref mut snapshot) = replay_kv {
                    snapshot.push((key_raw, value_raw));
                }
            }
            self.last_prefill_cold_seal_seconds = seal_started.elapsed().as_secs_f64();
        } else {
            self.last_prefill_cold_seal_seconds = 0.0;
        }
        // OPTIONAL GPU-ONLY hot tier for exact epsilon=0 decoding:
        // canonical ColdStore records and original index were created
        // above, so hot V never replaces the authoritative BF16 store.
        // All prompt V bytes are already in device temporary buffers:
        // promoting is GPU->GPU copy with ZERO host/cold reads.
        if let Some(hot) = self.hot_values.as_mut() {
            if prompt_ids.len() <= hot.capacity_tokens {
                let temporary = temp_values
                    .as_ref()
                    .context("hot V requires exact temporary BF16 prefill cache")?;
                let byte_count = prompt_ids.len() * kv_bytes;
                for (layer, src) in temporary.iter().enumerate() {
                    engine.copy(src, &mut hot.values[layer], byte_count, 0)?;
                }
                // Don't drop source GPU buffers until copies completed.
                engine
                    .queue
                    .finish()
                    .context("GPU exact hot V promote barrier")?;
                hot.valid_prefix_tokens = prompt_ids.len();
                hot.prefetched_bytes = byte_count * temporary.len();
            } else {
                hot.valid_prefix_tokens = 0;
            }
        }
        // Dropping temporary BF16 V releases the prefill-only allocation.
        // The optional HOT tier remains bounded, while ColdStore still
        // owns the bit-exact original K/V and recovers on cache misses.
        drop(temp_values);
        let source_offset = (last_chunk_count - 1) * hidden_bytes;
        // SAFETY: Last batch tensor includes final prompt token. The
        // destination is existing single-token hidden0 BF16 workspace.
        unsafe {
            engine
                .queue
                .enqueue_copy_buffer(
                    &batch.hidden0,
                    &mut self.workspace.hidden0,
                    source_offset,
                    0,
                    hidden_bytes,
                    &[],
                )
                .context("GPU last prefill token hidden state copy")?;
        }
        engine.norm(
            &self.workspace.hidden0,
            &self.final_norm,
            &self.workspace.norm1,
            u32::try_from(hidden)?,
            eps,
        )?;
        engine.gemv(
            &self.lm_head,
            &self.workspace.norm1,
            &self.workspace.logits,
            hidden,
            vocab,
        )?;
        engine.argmax(
            &self.workspace.logits,
            &self.workspace.token_id,
            u32::try_from(vocab)?,
        )?;
        let token = engine.read_token(&self.workspace.token_id)?;
        let logits = if capture_logits {
            Some(engine.read_logits(&self.workspace.logits, vocab)?)
        } else {
            None
        };
        // Store the original BF16 terminal logits and the exact K/V
        // bytes already verified by ColdStore. Future cache hits compare
        // EXACT tokenizer IDs and SHA256 before any reuse, and recreate
        // ColdStore/ParallelContextAxis from these unmodified bytes.
        if let Some(kv) = replay_kv {
            ensure!(
                kv.len() == self.spec.num_hidden_layers,
                "exact GPU replay snapshot missing BF16 layer"
            );
            let original_logits =
                engine.read_bf16_bytes(&self.workspace.logits, vocab * BYTES_PER_BF16)?;
            let bytes =
                kv.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>() + original_logits.len();
            let digest =
                VerifiedGpuPromptReplay::calculate_digest(prompt_ids, &kv, &original_logits, token);
            self.gpu_prompt_replay = Some(VerifiedGpuPromptReplay {
                ids: prompt_ids.to_vec(),
                kv,
                last_logits_bf16: original_logits,
                argmax: token,
                sha256: digest,
                bytes,
            });
        }
        self.position = prompt_ids.len();
        Ok(GpuStep {
            argmax_token: token,
            position: prompt_ids.len() - 1,
            forward_seconds: start.elapsed().as_secs_f64(),
            logits,
        })
    }
}

// Each layer's causal batch uses the SAME original matrices and BF16
// rounding operations as the old single-token GPU engine. No CPU neural
// arithmetic, no Python, no approximate KV values.
#[allow(clippy::too_many_arguments)]
fn run_batch_layer(
    engine: &GpuEngine,
    kernels: &BatchKernels,
    spec: &ModelSpec,
    layer: &GpuLayer,
    b: &BatchBuffers,
    key_cache: &mut Buffer<u8>,
    value_cache: &mut Buffer<u8>,
    prior: usize,
    tokens: usize,
    hidden: usize,
    middle: usize,
    eps: f32,
    rope: RopeKernelConfig,
) -> Result<()> {
    let geom = spec.attention;
    kernels.norm(
        engine,
        &b.hidden0,
        &layer.input_norm,
        &b.norm1,
        hidden,
        tokens,
        eps,
    )?;
    if kernels.fusion_mask & 1 != 0 {
        kernels.project_qkv8(
            engine,
            (&layer.q, &layer.k, &layer.v),
            (&b.norm1, &b.query, &b.key, &b.value),
            (
                hidden,
                geom.query_elements(),
                geom.kv_elements_per_token(),
                tokens,
            ),
        )?;
    } else {
        kernels.gemv(
            engine,
            &layer.q,
            &b.norm1,
            &b.query,
            hidden,
            geom.query_elements(),
            tokens,
        )?;
        kernels.gemv(
            engine,
            &layer.k,
            &b.norm1,
            &b.key,
            hidden,
            geom.kv_elements_per_token(),
            tokens,
        )?;
        kernels.gemv(
            engine,
            &layer.v,
            &b.norm1,
            &b.value,
            hidden,
            geom.kv_elements_per_token(),
            tokens,
        )?;
    }
    kernels.rope(
        engine,
        &b.query,
        &b.query_rope,
        tokens,
        geom.query_heads,
        geom.head_dim,
        prior,
        rope,
    )?;
    kernels.rope(
        engine,
        &b.key,
        &b.key_rope,
        tokens,
        geom.kv_heads,
        geom.head_dim,
        prior,
        rope,
    )?;
    let kv_bytes = geom.kv_elements_per_token() * 2;
    let size = tokens * kv_bytes;
    let offset = prior * kv_bytes;
    // SAFETY: Buffers are large enough for the entire prompt:
    // keys original Workspace uses configured max_context_tokens;
    // V original Workspace or short-lived temporary prefill V.
    unsafe {
        engine
            .queue
            .enqueue_copy_buffer(&b.key_rope, key_cache, 0, offset, size, &[])
            .context("GPU batch copy exact BF16 keys")?;
        engine
            .queue
            .enqueue_copy_buffer(&b.value, value_cache, 0, offset, size, &[])
            .context("GPU batch copy exact BF16 values")?;
    }
    kernels.attention(
        engine,
        &b.query_rope,
        key_cache,
        value_cache,
        &b.scores,
        &b.attention,
        tokens,
        prior,
        geom,
    )?;
    kernels.gemv(
        engine,
        &layer.o,
        &b.attention,
        &b.attn_o,
        geom.query_elements(),
        hidden,
        tokens,
    )?;
    engine.add(&b.hidden0, &b.attn_o, &b.hidden1, tokens * hidden)?;
    kernels.norm(
        engine,
        &b.hidden1,
        &layer.post_norm,
        &b.norm2,
        hidden,
        tokens,
        eps,
    )?;
    if kernels.fusion_mask & 2 != 0 {
        kernels.project_swiglu8(
            engine,
            (&layer.gate, &layer.up),
            &b.norm2,
            &b.activation,
            (hidden, middle, tokens),
        )?;
    } else {
        kernels.gemv(
            engine,
            &layer.gate,
            &b.norm2,
            &b.gate,
            hidden,
            middle,
            tokens,
        )?;
        kernels.gemv(engine, &layer.up, &b.norm2, &b.up, hidden, middle, tokens)?;
        engine.swiglu(&b.gate, &b.up, &b.activation, tokens * middle)?;
    }
    kernels.gemv(
        engine,
        &layer.down,
        &b.activation,
        &b.down,
        middle,
        hidden,
        tokens,
    )?;
    engine.add(&b.hidden1, &b.down, &b.hidden0, tokens * hidden)?;
    Ok(())
}
