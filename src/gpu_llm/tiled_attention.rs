//! Optional kernels sharing the existing GPU context, buffers and ColdStore.
//! No automatic activation, CPU simulation, hidden quantization or new KV store.
use super::GpuEngine;
use crate::{
    attention_kernel::{AttentionKernelConfig, AttentionKernelMode},
    portable::AttentionGeometry,
};
use anyhow::{Context, Result, ensure};
use opencl3::{
    device::Device,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, ClMem},
    program::Program,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) struct TiledAttention {
    pub(crate) config: AttentionKernelConfig,
    geometry: AttentionGeometry,
    _program: Program,
    kernel: Kernel,
    merge: Kernel,
    partials: Buffer<f32>,
    scratch_bytes: usize,
    key: String,
    local_bytes: u64,
    preferred_multiple: usize,
    local_memory_type: u32,
    calls: AtomicU64,
    split_calls: AtomicU64,
}
impl TiledAttention {
    pub(crate) fn build(
        engine: &GpuEngine,
        config: AttentionKernelConfig,
        geometry: AttentionGeometry,
    ) -> Result<Self> {
        config.validate()?;
        AttentionGeometry::new(geometry.query_heads, geometry.kv_heads, geometry.head_dim)?;
        ensure!(
            config.mode != AttentionKernelMode::Reference,
            "reference does not compile tiled kernels"
        );
        ensure!(
            geometry.query_heads > 0
                && geometry.kv_heads > 0
                && geometry.query_heads.is_multiple_of(geometry.kv_heads),
            "invalid grouped attention geometry"
        );
        let estimate = config.local_bytes(geometry.head_dim)?;
        let device = Device::new(
            *engine
                .context
                .devices()
                .first()
                .context("GPU context has no device")?,
        );
        ensure!(
            config.work_group <= device.max_work_group_size()?
                && config.work_group <= device.max_work_item_sizes()?[0],
            "attention workgroup exceeds device limit"
        );
        ensure!(
            estimate as u64 <= device.local_mem_size()?,
            "attention tile exceeds device local-memory limit"
        );
        let source = format!(
            "{}\n{}",
            include_str!("../../assets/gpu_llm.cl"),
            include_str!("../../assets/tiled_attention.cl")
        );
        let options = format!(
            "-cl-std=CL1.2 -DAT_D={} -DAT_BM={} -DAT_BN={} -DAT_WG={}",
            geometry.head_dim, config.block_queries, config.block_keys, config.work_group
        );
        let mut h = Sha256::new();
        h.update(b"mslim-tiled-attention-v1");
        h.update(source.as_bytes());
        h.update(options.as_bytes());
        h.update(engine.calibration_device_key.as_bytes());
        let key = hex::encode(h.finalize());
        let root = std::env::var_os("MSLIM_GPU_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "deploy/local-production/runtime/artifacts/gpu".into());
        let path = root.join(format!("tiled-{key}.bin"));
        let cached = fs::read(&path).ok().and_then(|binary| {
            Program::create_and_build_from_binary(&engine.context, &[binary.as_slice()], &options)
                .ok()
        });
        let program = if let Some(program) = cached {
            program
        } else {
            let program = Program::create_and_build_from_source(&engine.context, &source, &options)
                .map_err(anyhow::Error::msg)
                .context("compile OpenCL C 1.2 tiled attention")?;
            // Cache is only an optimization. Atomic unique temporary files avoid writer races.
            if let Ok(binaries) = program.get_binaries()
                && let Some(binary) = binaries.first()
                && fs::create_dir_all(&root).is_ok()
                && let Ok(mut temp) = tempfile::NamedTempFile::new_in(&root)
                && temp.write_all(binary).is_ok()
            {
                let _ = temp.persist(&path);
            }
            program
        };
        let kernel = Kernel::create(&program, "mslim_attention_tiled")?;
        let merge = Kernel::create(&program, "mslim_attention_merge")?;
        for k in [&kernel, &merge] {
            ensure!(
                config.work_group <= k.get_work_group_size(device.id())?,
                "kernel workgroup resource limit exceeded"
            );
            ensure!(
                k.get_local_mem_size(device.id())? <= device.local_mem_size()?,
                "compiled kernel exceeds local-memory limit"
            );
        }
        let local_bytes = kernel.get_local_mem_size(device.id())?;
        let preferred_multiple = kernel.get_work_group_size_multiple(device.id())?;
        let scratch_bytes = config.decode_scratch_bytes(geometry.query_heads, geometry.head_dim)?;
        // OpenCL requires a valid memory argument even when the split branch is disabled.
        let partials = engine.alloc_scores((scratch_bytes / 4).max(1))?;
        Ok(Self {
            config,
            geometry,
            _program: program,
            kernel,
            merge,
            partials,
            scratch_bytes,
            key,
            local_bytes,
            preferred_multiple,
            local_memory_type: device.local_mem_type()?,
            calls: AtomicU64::new(0),
            split_calls: AtomicU64::new(0),
        })
    }

    pub(crate) fn diagnostics(&self) -> Value {
        json!({"settings":self.config,"program_key":self.key,
            "numeric_contract":if self.config.experimental(){"experimental_online_not_bitwise"}else{"reference_order_three_pass"},
            "local_bytes_per_group":self.local_bytes,"local_memory_type":self.local_memory_type,
            "preferred_work_group_multiple":self.preferred_multiple,
            "global_score_matrix_bytes":0,"split_scratch_bytes":self.scratch_bytes,
            "unused_memory_argument_bytes":if self.scratch_bytes==0 {4}else{0},
            "dispatches_since_configuration":self.calls.load(Ordering::Relaxed),
            "split_dispatches_since_configuration":self.split_calls.load(Ordering::Relaxed),
            "split_policy":"split only single-query decode; multi-query prefill uses online blocks",
            "fallback_policy":"nonresident ColdStore V retains the original verified attention route"})
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        engine: &GpuEngine,
        q: &Buffer<u8>,
        k: &Buffer<u8>,
        v: &Buffer<u8>,
        output: &Buffer<u8>,
        queries: usize,
        prior: usize,
        geometry: AttentionGeometry,
    ) -> Result<()> {
        ensure!(
            queries > 0 && queries <= 512,
            "attention query count must be 1..=512"
        );
        ensure!(
            geometry.query_heads == self.geometry.query_heads
                && geometry.kv_heads == self.geometry.kv_heads
                && geometry.head_dim == self.geometry.head_dim,
            "kernel specialization does not match geometry"
        );
        let history = prior
            .checked_add(queries)
            .context("attention history overflow")?;
        ensure!(
            history <= u32::MAX as usize - 64,
            "attention history exceeds indexing bound"
        );
        let qbytes = queries
            .checked_mul(geometry.query_elements())
            .and_then(|x| x.checked_mul(2))
            .context("query buffer overflow")?;
        let kvbytes = history
            .checked_mul(geometry.kv_elements_per_token())
            .and_then(|x| x.checked_mul(2))
            .context("KV buffer overflow")?;
        ensure!(
            q.size()? >= qbytes
                && output.size()? >= qbytes
                && k.size()? >= kvbytes
                && v.size()? >= kvbytes,
            "tiled attention buffer is smaller than its declared geometry"
        );
        ensure!(
            output.get() != q.get() && output.get() != k.get() && output.get() != v.get(),
            "tiled attention output must not alias read-only Q/K/V"
        );
        let splits = if self.config.mode == AttentionKernelMode::SplitKvExperimental && queries == 1
        {
            self.config.splits.min(history)
        } else {
            1
        };
        let groups = queries
            .div_ceil(self.config.block_queries)
            .checked_mul(geometry.query_heads)
            .context("attention launch overflow")?;
        // Bound work per enqueue on display-attached GPUs. Query/head groups are
        // independent, so splitting launches does not change any reduction order.
        let passes = if self.config.experimental() {
            1usize
        } else {
            3
        };
        let per_group = history
            .checked_mul(geometry.head_dim)
            .and_then(|n| n.checked_mul(self.config.block_queries))
            .and_then(|n| n.checked_mul(passes))
            .context("attention dispatch work overflow")?;
        const MAX_DISPATCH_FMAS: usize = 32 * 1024 * 1024;
        ensure!(
            per_group <= MAX_DISPATCH_FMAS,
            "attention group exceeds dispatch work budget; use fewer block_queries"
        );
        let group_batch = (MAX_DISPATCH_FMAS / per_group.max(1)).clamp(1, 64);
        let local = [self.config.work_group, 1];
        let (b, p, n, qh, kh, d, s) = (
            u32::try_from(queries)?,
            u32::try_from(prior)?,
            u32::try_from(history)?,
            u32::try_from(geometry.query_heads)?,
            u32::try_from(geometry.kv_heads)?,
            u32::try_from(geometry.head_dim)?,
            u32::try_from(splits)?,
        );
        let online = u32::from(self.config.experimental());
        // SAFETY: Geometry, checked byte sizes, kernel/device resources and launch products
        // are validated above. Inputs remain read-only. The existing queue is in-order;
        // all buffers outlive both enqueues and the split merge follows its producers.
        unsafe {
            for first_group in (0..groups).step_by(group_batch) {
                let launch_groups = (groups - first_group).min(group_batch);
                let global = [launch_groups * self.config.work_group, splits];
                let group_offset = u32::try_from(first_group)?;
                ExecuteKernel::new(&self.kernel)
                    .set_arg(q)
                    .set_arg(k)
                    .set_arg(v)
                    .set_arg(output)
                    .set_arg(&self.partials)
                    .set_arg(&b)
                    .set_arg(&p)
                    .set_arg(&n)
                    .set_arg(&qh)
                    .set_arg(&kh)
                    .set_arg(&d)
                    .set_arg(&online)
                    .set_arg(&s)
                    .set_arg(&group_offset)
                    .set_global_work_sizes(&global)
                    .set_local_work_sizes(&local)
                    .enqueue_nd_range(&engine.queue)
                    .context("dispatch tiled attention")?;
            }
            if splits > 1 {
                ExecuteKernel::new(&self.merge)
                    .set_arg(&self.partials)
                    .set_arg(output)
                    .set_arg(&qh)
                    .set_arg(&d)
                    .set_arg(&s)
                    .set_global_work_size(geometry.query_heads * self.config.work_group)
                    .set_local_work_size(self.config.work_group)
                    .enqueue_nd_range(&engine.queue)
                    .context("merge split-KV attention")?;
            }
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        if splits > 1 {
            self.split_calls.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "physical-conformance"))]
mod tests {
    use super::*;
    use crate::gpu_llm::{GpuLlm, WeightMode, prefill::BatchKernels};
    use crate::gpu_region_store::GpuRegionConfig;
    use half::bf16;
    use opencl3::types::CL_BLOCKING;
    use std::time::Instant;

    fn input(count: usize, salt: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|i| {
                let value = (((i.wrapping_mul(37) + salt * 13) % 257) as i32 - 128) as f32 / 128.0;
                bf16::from_f32(value).to_bits().to_le_bytes()
            })
            .collect()
    }
    fn upload(engine: &GpuEngine, data: &[u8]) -> Buffer<u8> {
        let mut b = engine.alloc_u8(data.len()).unwrap();
        // SAFETY: Allocation is data.len() bytes; blocking transfer keeps host bytes alive.
        unsafe {
            engine
                .queue
                .enqueue_write_buffer(&mut b, CL_BLOCKING, 0, data, &[])
                .unwrap();
        }
        b
    }
    fn max_error(a: &[u8], b: &[u8]) -> f32 {
        a.as_chunks::<2>()
            .0
            .iter()
            .zip(b.as_chunks::<2>().0)
            .map(|(x, y)| {
                let x = bf16::from_bits(u16::from_le_bytes(*x)).to_f32();
                let y = bf16::from_bits(u16::from_le_bytes(*y)).to_f32();
                assert!(x.is_finite() && y.is_finite());
                (x - y).abs()
            })
            .fold(0.0, f32::max)
    }
    fn exercise(
        engine: &GpuEngine,
        cfg: AttentionKernelConfig,
        g: AttentionGeometry,
        lengths: &[usize],
    ) {
        let tiled = TiledAttention::build(engine, cfg, g).unwrap();
        let reference = BatchKernels::load(engine, 1, false, 128, 0).unwrap();
        for &n in lengths {
            for queries in [1usize, n.min(5)] {
                let prior = n - queries;
                let qb = input(queries * g.query_elements(), 3);
                let kb = input(n * g.kv_elements_per_token(), 7);
                let vb = input(n * g.kv_elements_per_token(), 11);
                let q = upload(engine, &qb);
                let k = upload(engine, &kb);
                let v = upload(engine, &vb);
                let a = engine.alloc_u8(qb.len()).unwrap();
                let b = engine.alloc_u8(qb.len()).unwrap();
                let scores = engine.alloc_scores(queries * g.query_heads * n).unwrap();
                reference
                    .attention(engine, &q, &k, &v, Some(&scores), &a, queries, prior, g)
                    .unwrap();
                tiled
                    .run(engine, &q, &k, &v, &b, queries, prior, g)
                    .unwrap();
                let expected = engine.read_bf16_bytes(&a, qb.len()).unwrap();
                let actual = engine.read_bf16_bytes(&b, qb.len()).unwrap();
                let error = max_error(&expected, &actual);
                if !cfg.experimental() {
                    assert!(
                        expected == actual,
                        "strict mismatch {} n={n} q={queries} dim={} max_error={error}",
                        cfg.id(),
                        g.head_dim
                    );
                } else {
                    // Explicit test tolerance for these synthetic vectors, never a production quality promise.
                    assert!(
                        error <= 0.0078125,
                        "experimental test tolerance exceeded: {error}"
                    );
                }
                tiled
                    .run(engine, &q, &k, &v, &b, queries, prior, g)
                    .unwrap();
                assert_eq!(
                    actual,
                    engine.read_bf16_bytes(&b, qb.len()).unwrap(),
                    "repeatability failed"
                );
                assert_eq!(
                    kb,
                    engine.read_bf16_bytes(&k, kb.len()).unwrap(),
                    "K was modified"
                );
                assert_eq!(
                    vb,
                    engine.read_bf16_bytes(&v, vb.len()).unwrap(),
                    "V was modified"
                );
                eprintln!(
                    "TILED_GPU_CASE {}",
                    json!({"config":cfg,"head_dim":g.head_dim,"qheads":g.query_heads,
                    "kvheads":g.kv_heads,"history":n,"queries":queries,"max_error":error,
                    "bit_equal":expected==actual,"repeat_equal":true,"kv_unchanged":true})
                );
            }
        }
        eprintln!("TILED_GPU_RESOURCES {}", tiled.diagnostics());
    }
    #[test]
    #[ignore = "requires physical OpenCL GPU; no checkpoint or CPU substitute"]
    fn physical_tiled_strict_reference_bits_and_causal_tails() {
        let engine = GpuEngine::new(false).unwrap();
        for (dim, qh, kh, bm, bn, wg) in [
            (64, 4, 4, 1, 32, 32),
            (128, 8, 2, 1, 64, 64),
            (128, 4, 1, 2, 64, 64),
            (128, 4, 1, 4, 32, 128),
            (80, 4, 2, 2, 32, 32),
            (256, 4, 1, 1, 64, 128),
        ] {
            let cfg = AttentionKernelConfig {
                mode: AttentionKernelMode::StrictTiled,
                block_queries: bm,
                block_keys: bn,
                work_group: wg,
                splits: 1,
            };
            exercise(
                &engine,
                cfg,
                AttentionGeometry::new(qh, kh, dim).unwrap(),
                &[1, 2, 31, 33, 63, 65, 129, 375, 2049, 4097],
            );
        }
    }
    #[test]
    #[ignore = "requires physical OpenCL GPU; experimental output is not a bitwise contract"]
    fn physical_tiled_online_and_split_fixed_order() {
        let engine = GpuEngine::new(false).unwrap();
        for (mode, splits, bm) in [
            (AttentionKernelMode::OnlineExperimental, 1, 2),
            (AttentionKernelMode::SplitKvExperimental, 8, 1),
        ] {
            let cfg = AttentionKernelConfig {
                mode,
                block_queries: bm,
                splits,
                ..Default::default()
            };
            exercise(
                &engine,
                cfg,
                AttentionGeometry::new(8, 2, 128).unwrap(),
                &[1, 2, 7, 33, 65, 129, 375, 2049, 4097],
            );
        }
    }
    #[test]
    #[ignore = "requires physical OpenCL GPU; bounded experimental smoke check"]
    fn physical_tiled_experimental_smoke() {
        let engine = GpuEngine::new(false).unwrap();
        for (mode, splits) in [
            (AttentionKernelMode::OnlineExperimental, 1),
            (AttentionKernelMode::SplitKvExperimental, 8),
        ] {
            let cfg = AttentionKernelConfig {
                mode,
                splits,
                ..Default::default()
            };
            eprintln!("TILED_EXPERIMENT_SMOKE_BEGIN {}", cfg.id());
            exercise(
                &engine,
                cfg,
                AttentionGeometry::new(4, 1, 128).unwrap(),
                &[1, 17, 65],
            );
        }
    }
    #[test]
    #[ignore = "requires physical OpenCL GPU"]
    fn physical_tiled_rejects_bad_geometry_and_buffers() {
        let e = GpuEngine::new(false).unwrap();
        let g = AttentionGeometry::new(4, 1, 128).unwrap();
        let cfg = AttentionKernelConfig {
            mode: AttentionKernelMode::StrictTiled,
            ..Default::default()
        };
        let plan = TiledAttention::build(&e, cfg, g).unwrap();
        let b = e.alloc_u8(2).unwrap();
        assert!(plan.run(&e, &b, &b, &b, &b, 1, 0, g).is_err());
        assert!(plan.run(&e, &b, &b, &b, &b, 0, 0, g).is_err());
        assert!(plan.run(&e, &b, &b, &b, &b, 1, usize::MAX, g).is_err());
        assert!(
            plan.run(
                &e,
                &b,
                &b,
                &b,
                &b,
                1,
                0,
                AttentionGeometry::new(4, 1, 64).unwrap()
            )
            .is_err()
        );
    }
    #[test]
    #[ignore = "requires configured real BF16 checkpoint and physical GPU"]
    fn physical_tiled_full_model_strict_logits_kv_and_reset() {
        let model = std::env::var("MSLIM_TEST_MODEL").expect("actual checkpoint required");
        let mut runtime = GpuLlm::load_with_gpu_region(
            &model,
            256,
            WeightMode::Resident,
            GpuRegionConfig::default(),
        )
        .unwrap();
        runtime.set_batch_prefill_tokens(Some(16)).unwrap();
        runtime.set_prefill_weight_reuse_tokens(8).unwrap();
        runtime.set_exact_hot_v_tokens(256).unwrap();
        runtime.set_verified_gpu_prompt_replay(0).unwrap();
        let cfg = AttentionKernelConfig {
            mode: AttentionKernelMode::StrictTiled,
            block_queries: 2,
            ..Default::default()
        };
        for length in [8usize, 17, 65] {
            let ids = (0..length)
                .map(|i| 100 + (i % 31) as u32)
                .collect::<Vec<_>>();
            runtime.reset_session().unwrap();
            runtime
                .set_attention_kernel(Default::default(), false)
                .unwrap();
            let baseline = runtime.generate(&ids, 3, true).unwrap();
            let sha = runtime.canonical_kv_sha256().unwrap();
            runtime.reset_session().unwrap();
            runtime.set_attention_kernel(cfg, false).unwrap();
            let actual = runtime.generate(&ids, 3, true).unwrap();
            assert_eq!(baseline.tokens, actual.tokens);
            assert_eq!(baseline.steps.len(), actual.steps.len());
            for (a, b) in baseline.steps.iter().zip(&actual.steps) {
                let a = a.logits.as_ref().unwrap();
                let b = b.logits.as_ref().unwrap();
                assert_eq!(a.len(), b.len());
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "full model bitwise logits mismatch at context {length}"
                );
            }
            assert_eq!(sha, runtime.canonical_kv_sha256().unwrap());
            let diag = runtime.attention_kernel_diagnostics();
            assert!(diag["dispatches_since_configuration"].as_u64().unwrap() > 0);
            eprintln!(
                "TILED_FULL_MODEL_PASS {}",
                json!({"tokens":length,"all_logit_bits_equal":true,"kv_sha_equal":true,"diagnostics":diag})
            );
        }
    }
    #[test]
    #[ignore = "physical GPU benchmark; timings must not be treated as model-level speedups"]
    fn physical_tiled_attention_timing_report() {
        let e = GpuEngine::new(false).unwrap();
        let g = AttentionGeometry::new(32, 8, 128).unwrap();
        let reference = BatchKernels::load(&e, 1, false, 128, 0).unwrap();
        let configs = [
            AttentionKernelConfig::default(),
            AttentionKernelConfig {
                mode: AttentionKernelMode::StrictTiled,
                block_queries: 2,
                ..Default::default()
            },
            AttentionKernelConfig {
                mode: AttentionKernelMode::OnlineExperimental,
                block_queries: 2,
                ..Default::default()
            },
            AttentionKernelConfig {
                mode: AttentionKernelMode::SplitKvExperimental,
                splits: 8,
                ..Default::default()
            },
        ];
        let mut reports = Vec::new();
        for &(queries, prior) in &[(1usize, 1023usize), (16, 1008), (1, 4095), (8, 4088)] {
            let n = queries + prior;
            eprintln!("TILED_TIMING_CASE_BEGIN queries={queries} history={n}");
            let qb = input(queries * g.query_elements(), 3);
            let q = upload(&e, &qb);
            let k = upload(&e, &input(n * g.kv_elements_per_token(), 7));
            let v = upload(&e, &input(n * g.kv_elements_per_token(), 11));
            let a = e.alloc_u8(qb.len()).unwrap();
            let scores = e.alloc_scores(queries * g.query_heads * n).unwrap();
            let plans = configs
                .iter()
                .map(|&c| {
                    if c.mode == AttentionKernelMode::Reference {
                        None
                    } else {
                        Some(TiledAttention::build(&e, c, g).unwrap())
                    }
                })
                .collect::<Vec<_>>();
            let mut samples = vec![Vec::new(); configs.len()];
            for round in 0..9 {
                // Rotate variants to reduce order/warmth bias; first two rounds are warmup.
                for slot in 0..configs.len() {
                    let i = (round + slot) % configs.len();
                    eprintln!(
                        "TILED_TIMING_BEGIN q={} history={} round={} config={}",
                        queries,
                        n,
                        round,
                        configs[i].id()
                    );
                    let start = Instant::now();
                    if let Some(p) = &plans[i] {
                        p.run(&e, &q, &k, &v, &a, queries, prior, g).unwrap();
                    } else {
                        reference
                            .attention(&e, &q, &k, &v, Some(&scores), &a, queries, prior, g)
                            .unwrap();
                    }
                    e.queue.finish().unwrap();
                    if round >= 2 {
                        samples[i].push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            for (i, mut samples) in samples.into_iter().enumerate() {
                samples.sort_by(f64::total_cmp);
                reports.push(json!({"queries":queries,"history":n,"config":configs[i],"samples_ms":samples,
                    "median_ms":samples[samples.len()/2],"reference_score_bytes":queries*g.query_heads*n*4,
                    "tiled_resources":plans[i].as_ref().map(|p|p.diagnostics())}));
            }
        }
        eprintln!(
            "TILED_TIMING_REPORT {}",
            json!({"device":e.name,"scope":"attention only, deterministic synthetic BF16 inputs, no model throughput claim","cases":reports})
        );
    }
}
