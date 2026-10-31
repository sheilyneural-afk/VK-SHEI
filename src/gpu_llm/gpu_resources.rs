//! GPU context, command queue, and device resource management.
//!
//! `GpuEngine` owns the OpenCL context, compiled program, and loaded kernels.
//! `impl GpuWeight`, `GpuLayer`, and `Workspace` upload safetensors weights to
//! device memory and allocate working buffers.
//!
//! All `unsafe` OpenCL C-API calls are concentrated here. Hardware selection
//! uses `get_all_devices(CL_DEVICE_TYPE_GPU)` at runtime — no vendor is
//! hard-coded.

use anyhow::{Context, Result, ensure};
use opencl3::{
    command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue},
    context::Context as ClContext,
    device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices},
    kernel::ExecuteKernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    program::Program,
    types::CL_BLOCKING,
};
use std::{
    fs, ptr,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(feature = "hip-backend")]
use crate::native_hip::{HipRuntime, HipWeight};
#[cfg(feature = "hip-backend")]
use opencl3::memory::ClMem;

use super::kernels::{
    CheckpointShards, GpuKv, GpuLayer, GpuWeight, LlmKernels, LodStage, MicroHotBuffers,
    MicroSelectedDevice, RopeKernelConfig, Workspace,
};
use crate::gpu_region_store::{GpuRegionConfig, GpuRegionStore};
use crate::model_config::ModelSpec;
use crate::portable::AttentionGeometry;
use safetensors::{Dtype, tensor::TensorView};
use sha2::{Digest, Sha256};

const LOCAL_WORK_SIZE: usize = 128;
const BYTES_PER_BF16: usize = 2;

pub(crate) struct GpuEngine {
    pub(crate) context: ClContext,
    pub(crate) queue: CommandQueue,
    // Keep program loaded while kernels are in use.
    pub(crate) _program: Program,
    pub(crate) kernels: LlmKernels,
    pub(crate) tiled_attention: Option<super::tiled_attention::TiledAttention>,
    pub name: String,
    pub reported_gpu_memory: u64,
    pub(crate) calibration_device_key: String,
    pub max_gpu_allocation: u64,
    pub(crate) gemv_unroll4_enabled: AtomicBool,
    pub(crate) fused_decode_enabled: AtomicBool,
    pub(crate) exact_hot_attention_enabled: AtomicBool,
    #[cfg(feature = "hip-backend")]
    pub(crate) hip: Option<HipRuntime>,
}

impl GpuEngine {
    pub(crate) fn new(hip_enabled: bool) -> Result<Self> {
        #[cfg(not(feature = "hip-backend"))]
        ensure!(!hip_enabled, "native HIP support was not compiled");
        #[cfg(feature = "hip-backend")]
        let hip = if hip_enabled {
            Some(HipRuntime::new()?)
        } else {
            None
        };
        let filter = std::env::var("MSLIM_GPU_DEVICE")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let devices = get_all_devices(CL_DEVICE_TYPE_GPU).context("no GPU devices available")?;
        let device = devices
            .into_iter()
            .map(Device::new)
            .find(|device| {
                let name = device.name().unwrap_or_default().to_lowercase();
                let vendor = device.vendor().unwrap_or_default().to_lowercase();
                filter.is_empty() || name.contains(&filter) || vendor.contains(&filter)
            })
            .context("no matching GPU found")?;
        ensure!(
            device.dev_type()? & CL_DEVICE_TYPE_GPU != 0,
            "selected device is not a physical GPU"
        );
        ensure!(
            device.endian_little()?,
            "GPU must be little-endian for BF16"
        );
        let name = device.name()?;
        let reported_gpu_memory = device.global_mem_size()?;
        let max_gpu_allocation = device.max_mem_alloc_size()?;
        let context = ClContext::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;
        let shader = include_str!("../../assets/gpu_llm.cl");
        let build_options = "-cl-std=CL1.2";
        let mut key_hasher = Sha256::new();
        key_hasher.update(shader.as_bytes());
        key_hasher.update(build_options.as_bytes());
        key_hasher.update(name.as_bytes());
        key_hasher.update(device.vendor()?.as_bytes());
        key_hasher.update(device.driver_version()?.as_bytes());
        key_hasher.update(device.version()?.as_bytes());
        let cache_key = hex::encode(key_hasher.finalize());
        let cache_root = std::env::var_os("MSLIM_GPU_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("deploy/local-production/runtime/artifacts/gpu")
            });
        let cache_path = cache_root.join(format!("{cache_key}.bin"));
        let program = if let Ok(binary) = fs::read(&cache_path) {
            let binaries: [&[u8]; 1] = [&binary];
            Program::create_and_build_from_binary(&context, &binaries, build_options)
                .or_else(|_| {
                    Program::create_and_build_from_source(&context, shader, build_options)
                        .map_err(anyhow::Error::msg)
                })
                .context("load or compile cached GPU program")?
        } else {
            let program = Program::create_and_build_from_source(&context, shader, build_options)
                .map_err(anyhow::Error::msg)
                .context("GPU full-LLM shader compilation failed")?;
            if let Ok(binaries) = program.get_binaries()
                && let Some(binary) = binaries.first()
            {
                let _ = fs::create_dir_all(&cache_root);
                let temporary = cache_path.with_extension("tmp");
                if fs::write(&temporary, binary).is_ok() {
                    let _ = fs::rename(temporary, &cache_path);
                }
            }
            program
        };
        let kernels = LlmKernels::load(&program)?;
        Ok(Self {
            context,
            queue,
            _program: program,
            kernels,
            tiled_attention: None,
            name,
            reported_gpu_memory,
            calibration_device_key: cache_key,
            max_gpu_allocation,
            gemv_unroll4_enabled: AtomicBool::new(false),
            fused_decode_enabled: AtomicBool::new(false),
            exact_hot_attention_enabled: AtomicBool::new(false),
            #[cfg(feature = "hip-backend")]
            hip,
        })
    }

    pub(crate) fn alloc_u8(&self, bytes: usize) -> Result<Buffer<u8>> {
        ensure!(
            bytes > 0 && bytes as u64 <= self.max_gpu_allocation,
            "GPU allocation {bytes} exceeds limit"
        );
        // SAFETY: The context is alive throughout the lifetime of the buffer.
        unsafe {
            Buffer::create(&self.context, CL_MEM_READ_WRITE, bytes, ptr::null_mut())
                .context("GPU memory allocation failed")
        }
    }

    pub(crate) fn alloc_scores(&self, count: usize) -> Result<Buffer<f32>> {
        ensure!(
            count > 0 && (count as u64) * 4 <= self.max_gpu_allocation,
            "GPU score buffer too large"
        );
        // SAFETY: The context is alive throughout the lifetime of the buffer.
        unsafe {
            Buffer::create(&self.context, CL_MEM_READ_WRITE, count, ptr::null_mut())
                .context("GPU score allocation failed")
        }
    }

    pub(crate) fn alloc_indices(&self, count: usize) -> Result<Buffer<u32>> {
        ensure!(
            count > 0 && (count as u64) * 4 <= self.max_gpu_allocation,
            "GPU LOD index staging too large"
        );
        // SAFETY: Context valid for the lifetime of the returned buffer.
        unsafe {
            Buffer::create(&self.context, CL_MEM_READ_WRITE, count, ptr::null_mut())
                .context("GPU LOD index allocation failed")
        }
    }

    pub(crate) fn alloc_result(&self) -> Result<Buffer<u32>> {
        // SAFETY: The context is alive throughout the lifetime of the buffer.
        unsafe {
            Buffer::create(&self.context, CL_MEM_READ_WRITE, 1, ptr::null_mut())
                .context("GPU result allocation failed")
        }
    }

    pub(crate) fn copy(
        &self,
        source: &Buffer<u8>,
        destination: &mut Buffer<u8>,
        size: usize,
        dest_byte_offset: usize,
    ) -> Result<()> {
        // SAFETY: Source and destination are valid OpenCL buffers with
        // sufficient capacity (validated by the executor's shape constraints).
        unsafe {
            self.queue
                .enqueue_copy_buffer(source, destination, 0, dest_byte_offset, size, &[])
                .context("GPU BF16 buffer copy failed")?;
        }
        Ok(())
    }

    pub(crate) fn transfer(&self, destination: &mut Buffer<u8>, bytes: &[u8]) -> Result<()> {
        // SAFETY: CL_BLOCKING retains bytes until DMA/transfer completes.
        unsafe {
            self.queue
                .enqueue_write_buffer(destination, CL_BLOCKING, 0, bytes, &[])
                .context("GPU upload of original BF16 bytes failed")?;
        }
        Ok(())
    }

    pub(crate) fn read_token(&self, source: &Buffer<u32>) -> Result<u32> {
        let mut token = [0u32; 1];
        // SAFETY: CL_BLOCKING retains the host buffer for the driver call.
        unsafe {
            self.queue
                .enqueue_read_buffer(source, CL_BLOCKING, 0, &mut token, &[])
                .context("GPU argmax readback failed")?;
        }
        Ok(token[0])
    }

    pub(crate) fn read_logits(&self, source: &Buffer<u8>, vocab: usize) -> Result<Vec<f32>> {
        let mut bits = vec![0u8; vocab * 2];
        // SAFETY: CL_BLOCKING retains the host buffer.
        unsafe {
            self.queue
                .enqueue_read_buffer(source, CL_BLOCKING, 0, &mut bits, &[])
                .context("GPU logits BF16 readback failed")?;
        }
        Ok(bits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect())
    }

    pub(crate) fn embed(
        &self,
        weight: &GpuWeight,
        output: &Buffer<u8>,
        token_id: u32,
        hidden_size: u32,
    ) -> Result<()> {
        // SAFETY: Kernel signature and number/type of args match OpenCL source.
        unsafe {
            ExecuteKernel::new(&self.kernels.embed)
                .set_arg(&weight.data)
                .set_arg(output)
                .set_arg(&token_id)
                .set_arg(&hidden_size)
                .set_global_work_size(hidden_size as usize)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 embedding failed")?;
        }
        Ok(())
    }

    /// HIP native BF16 GEMM: row-major [rows,inner] weights and
    /// [tokens,inner] activations; OpenCL buffers host-staged only at
    /// layer matrix boundaries. All GEMM arithmetic executes on real
    /// AMD gfx1103 hipBLASLt matrix hardware. No CPU neural math.
    #[cfg(feature = "hip-backend")]
    pub(crate) fn hip_matrix(
        &self,
        weight: &HipWeight,
        input: &Buffer<u8>,
        output: &Buffer<u8>,
        inner: usize,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let native = self.hip.as_ref().context("no native HIP BF16 context")?;
        let input_bytes = inner
            .checked_mul(tokens)
            .and_then(|n| n.checked_mul(2))
            .context("HIP input size overflow")?;
        let output_bytes = rows
            .checked_mul(tokens)
            .and_then(|n| n.checked_mul(2))
            .context("HIP output size overflow")?;
        let mut host_in = vec![0u8; input_bytes];
        // SAFETY: OpenCL buffer is valid, this is a blocking readback of
        // queued BF16 activations, with no CPU floating point compute.
        unsafe {
            self.queue
                .enqueue_read_buffer(input, CL_BLOCKING, 0, &mut host_in, &[])
                .context("GPU to native HIP matrix activation")?;
        }
        let mut host_out = vec![0u8; output_bytes];
        native.gemm_bf16(weight, &host_in, &mut host_out, inner, rows, tokens)?;
        // SAFETY: All GPU OpenCL buffers are owned by the inference
        // context. Direct OpenCL C API accepts the existing buffer handle;
        // CL_TRUE blocks until the BF16 output upload completes.
        #[link(name = "OpenCL")]
        unsafe extern "C" {
            pub(crate) fn clEnqueueWriteBuffer(
                command_queue: *mut std::ffi::c_void,
                buffer: *mut std::ffi::c_void,
                blocking_write: u32,
                offset: usize,
                cb: usize,
                ptr: *const std::ffi::c_void,
                num_events_in_wait_list: u32,
                event_wait_list: *const std::ffi::c_void,
                event: *mut std::ffi::c_void,
            ) -> i32;
        }
        // SAFETY: ABI of OpenCL clEnqueueWriteBuffer, same cl_mem and
        // cl_command_queue types used by opencl3. Blocking transfer.
        let rc = unsafe {
            clEnqueueWriteBuffer(
                self.queue.get(),
                output.get(),
                1,
                0,
                host_out.len(),
                host_out.as_ptr().cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        ensure!(
            rc == 0,
            "GPU upload from native HIP BF16 matrix failed ({rc})"
        );
        Ok(())
    }

    pub(crate) fn gemv(
        &self,
        weight: &GpuWeight,
        input: &Buffer<u8>,
        output: &Buffer<u8>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<()> {
        #[cfg(feature = "hip-backend")]
        if let Some(ref hip) = weight.hip {
            return self.hip_matrix(hip, input, output, in_dim, out_dim, 1);
        }
        let in_dim_u32 = u32::try_from(in_dim)?;
        let out_dim_u32 = u32::try_from(out_dim)?;
        let global = out_dim
            .checked_mul(LOCAL_WORK_SIZE)
            .context("GPU GEMV workgroup count overflow")?;
        // SAFETY: Matrices are validated BF16 [out_dim,in_dim]; the kernel
        // allocates one workgroup of exactly 128 threads per output row.
        unsafe {
            ExecuteKernel::new(if self.gemv_unroll4_enabled.load(Ordering::Relaxed) {
                &self.kernels.gemv_unroll4
            } else {
                &self.kernels.gemv
            })
            .set_arg(&weight.data)
            .set_arg(input)
            .set_arg(output)
            .set_arg(&in_dim_u32)
            .set_arg(&out_dim_u32)
            .set_global_work_size(global)
            .set_local_work_size(LOCAL_WORK_SIZE)
            .enqueue_nd_range(&self.queue)
            .with_context(|| format!("GPU BF16 GEMV [{out_dim} x {in_dim}] failed"))?;
        }
        Ok(())
    }

    /// Fused three original BF16 projections; no weight quantization,
    /// no skipped Q/K/V values, same per-row summation order as GEMV.
    pub(crate) fn project_fused_qkv(
        &self,
        weights: (&GpuWeight, &GpuWeight, &GpuWeight),
        buffers: (&Buffer<u8>, &Buffer<u8>, &Buffer<u8>, &Buffer<u8>),
        shape: (usize, usize, usize),
    ) -> Result<()> {
        let (q, k, v) = weights;
        let (input, outq, outk, outv) = buffers;
        let (hidden, qrows, kvrows) = shape;
        ensure!(
            qrows >= kvrows && hidden > 0 && kvrows > 0,
            "fused GPU QKV projection shape invalid"
        );
        #[cfg(feature = "hip-backend")]
        ensure!(
            q.hip.is_none() && k.hip.is_none() && v.hip.is_none(),
            "fused GPU QKV cannot read HIP-only BF16 weights"
        );
        let (d, n, m) = (
            u32::try_from(hidden)?,
            u32::try_from(qrows)?,
            u32::try_from(kvrows)?,
        );
        // SAFETY: All matrix dimensions were validated by original
        // BF16 safetensors loader, NDRanges 128 lanes per output row.
        unsafe {
            ExecuteKernel::new(&self.kernels.fused_qkv)
                .set_arg(&q.data)
                .set_arg(&k.data)
                .set_arg(&v.data)
                .set_arg(input)
                .set_arg(outq)
                .set_arg(outk)
                .set_arg(outv)
                .set_arg(&d)
                .set_arg(&n)
                .set_arg(&m)
                .set_global_work_size(qrows * LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 fused QKV projection failed")?;
        }
        Ok(())
    }

    pub(crate) fn project_fused_swiglu(
        &self,
        weights: (&GpuWeight, &GpuWeight),
        input: &Buffer<u8>,
        output: &Buffer<u8>,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        let (gate, up) = weights;
        #[cfg(feature = "hip-backend")]
        ensure!(
            gate.hip.is_none() && up.hip.is_none(),
            "GPU fused SwiGLU cannot access HIP-owned BF16 weights"
        );
        let (d, m) = (u32::try_from(hidden)?, u32::try_from(intermediate)?);
        // SAFETY: Original weights and the single-token BF16 normalized
        // input remain allocated through enqueue; kernel exactly reduces
        // 128 lanes per MLP output row and rounds each original stage.
        unsafe {
            ExecuteKernel::new(&self.kernels.fused_swiglu)
                .set_arg(&gate.data)
                .set_arg(&up.data)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&d)
                .set_arg(&m)
                .set_global_work_size(intermediate * LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 fused SwiGLU GEMV failed")?;
        }
        Ok(())
    }

    pub(crate) fn norm(
        &self,
        input: &Buffer<u8>,
        weight: &GpuWeight,
        output: &Buffer<u8>,
        hidden: u32,
        eps: f32,
    ) -> Result<()> {
        // SAFETY: Workgroup size is 128 as declared by kernel attribute.
        unsafe {
            ExecuteKernel::new(&self.kernels.norm)
                .set_arg(input)
                .set_arg(&weight.data)
                .set_arg(output)
                .set_arg(&hidden)
                .set_arg(&eps)
                .set_global_work_size(LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 RMSNorm failed")?;
        }
        Ok(())
    }

    pub(crate) fn rope(
        &self,
        input: &Buffer<u8>,
        output: &Buffer<u8>,
        heads: u32,
        head_dim: u32,
        position: u32,
        rope: RopeKernelConfig,
    ) -> Result<()> {
        // SAFETY: OpenCL kernel checks work index bounds.
        unsafe {
            ExecuteKernel::new(&self.kernels.rope)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&heads)
                .set_arg(&head_dim)
                .set_arg(&position)
                .set_arg(&rope.theta)
                .set_arg(&rope.llama3_scaling)
                .set_arg(&rope.factor)
                .set_arg(&rope.low_freq_factor)
                .set_arg(&rope.high_freq_factor)
                .set_arg(&rope.original_max_position)
                .set_global_work_size((heads as usize) * (head_dim as usize))
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 RoPE failed")?;
        }
        Ok(())
    }

    pub(crate) fn add(
        &self,
        left: &Buffer<u8>,
        right: &Buffer<u8>,
        output: &Buffer<u8>,
        count: usize,
    ) -> Result<()> {
        let count_u32 = u32::try_from(count)?;
        // SAFETY: OpenCL kernel checks work index bounds.
        unsafe {
            ExecuteKernel::new(&self.kernels.add)
                .set_arg(left)
                .set_arg(right)
                .set_arg(output)
                .set_arg(&count_u32)
                .set_global_work_size(count)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 residual add failed")?;
        }
        Ok(())
    }

    pub(crate) fn swiglu(
        &self,
        gate: &Buffer<u8>,
        up: &Buffer<u8>,
        output: &Buffer<u8>,
        count: usize,
    ) -> Result<()> {
        let count_u32 = u32::try_from(count)?;
        // SAFETY: OpenCL kernel checks work index bounds.
        unsafe {
            ExecuteKernel::new(&self.kernels.swiglu)
                .set_arg(gate)
                .set_arg(up)
                .set_arg(output)
                .set_arg(&count_u32)
                .set_global_work_size(count)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 SwiGLU failed")?;
        }
        Ok(())
    }

    pub(crate) fn attention(
        &self,
        query: &Buffer<u8>,
        kv: (&Buffer<u8>, &Buffer<u8>),
        scores: &Buffer<f32>,
        output: &Buffer<u8>,
        tokens: u32,
        geometry: AttentionGeometry,
    ) -> Result<()> {
        if let Some(tiled) = &self.tiled_attention {
            ensure!(tokens > 0, "empty tiled decode");
            return tiled.run(
                self,
                query,
                kv.0,
                kv.1,
                output,
                1,
                tokens as usize - 1,
                geometry,
            );
        }
        let (key, value) = kv;
        let qheads = u32::try_from(geometry.query_heads)?;
        let kvheads = u32::try_from(geometry.kv_heads)?;
        let dim = u32::try_from(geometry.head_dim)?;
        // SAFETY: Each shader uses valid matching memory and scalars; this
        // queue is in-order, so PV sees the completed softmax scores.
        unsafe {
            ExecuteKernel::new(&self.kernels.qk)
                .set_arg(query)
                .set_arg(key)
                .set_arg(scores)
                .set_arg(&tokens)
                .set_arg(&qheads)
                .set_arg(&kvheads)
                .set_arg(&dim)
                .set_global_work_size((tokens as usize) * (qheads as usize))
                .enqueue_nd_range(&self.queue)
                .context("GPU attention QK failed")?;
            ExecuteKernel::new(&self.kernels.softmax)
                .set_arg(scores)
                .set_arg(&tokens)
                .set_arg(&qheads)
                .set_global_work_size(qheads as usize)
                .enqueue_nd_range(&self.queue)
                .context("GPU attention softmax failed")?;
            ExecuteKernel::new(&self.kernels.pv)
                .set_arg(scores)
                .set_arg(value)
                .set_arg(output)
                .set_arg(&tokens)
                .set_arg(&qheads)
                .set_arg(&kvheads)
                .set_arg(&dim)
                .set_global_work_size((qheads as usize) * (dim as usize))
                .enqueue_nd_range(&self.queue)
                .context("GPU attention PV failed")?;
        }
        Ok(())
    }

    /// Sparse real Q·K GPU kernel, reached ONLY after original CPU
    /// micro-links/4-bit LSH priorities + BF16 K-box omitted-mass proof.
    /// All nonselected token scores become -INFINITY; original GPU
    /// softmax and BF16 PV operate on certified real selected K/V.
    pub(crate) fn micro_sparse_scores(
        &self,
        query: &Buffer<u8>,
        keys: &Buffer<u8>,
        scores: &Buffer<f32>,
        positions: &[usize],
        geometry: AttentionGeometry,
        total: usize,
    ) -> Result<()> {
        ensure!(
            !positions.is_empty() && positions.len() < total,
            "GPU sparse micro path has invalid selected K count"
        );
        ensure!(
            positions.windows(2).all(|w| w[0] < w[1]) && positions.iter().all(|&pos| pos < total),
            "GPU sparse micro selected token IDs not sorted/unique"
        );
        let count = geometry
            .query_heads
            .checked_mul(total)
            .context("GPU micro score scratch count overflow")?;
        let score_count = u32::try_from(count)?;
        let selected_count = u32::try_from(positions.len())?;
        let total_count = u32::try_from(total)?;
        let heads = u32::try_from(geometry.query_heads)?;
        let kvheads = u32::try_from(geometry.kv_heads)?;
        let dim = u32::try_from(geometry.head_dim)?;
        let ids = positions
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<u32>, _>>()?;
        let mut indices = self.alloc_indices(ids.len())?;
        // SAFETY: GPU score bounds are checked; all OpenCL indices
        // originate from the canonical original CPU region IDs and
        // were checked against the committed K length above.
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut indices, CL_BLOCKING, 0, &ids, &[])
                .context("upload certified GPU micro token positions")?;
            ExecuteKernel::new(&self.kernels.micro_clear)
                .set_arg(scores)
                .set_arg(&score_count)
                .set_global_work_size(count)
                .enqueue_nd_range(&self.queue)
                .context("initialize omitted GPU QK micro scores")?;
            ExecuteKernel::new(&self.kernels.micro_sparse_qk)
                .set_arg(query)
                .set_arg(keys)
                .set_arg(&indices)
                .set_arg(scores)
                .set_arg(&selected_count)
                .set_arg(&total_count)
                .set_arg(&heads)
                .set_arg(&kvheads)
                .set_arg(&dim)
                .set_global_work_size(positions.len() * geometry.query_heads)
                .enqueue_nd_range(&self.queue)
                .context("GPU real BF16 QK only for certified micro regions")?;
        }
        // Temporary indices buffer cannot be freed while QK still
        // references it; finish the queue before returning.
        self.queue
            .finish()
            .context("GPU micro sparse QK indices lifetime")?;
        Ok(())
    }

    /// One physical GPU BF16 attention kernel with QK, softmax and PV
    /// fused. Indices are RESIDENT in GPU device memory and may be
    /// reused across consultations. No K/V host read or cold store
    /// access can occur inside this operation.
    pub(crate) fn micro_hot_attention(
        &self,
        buffers: MicroHotBuffers<'_>,
        selection: (usize, usize),
        geometry: AttentionGeometry,
    ) -> Result<()> {
        let (query, keys, values, indices, output) = buffers;
        let (selected, total) = selection;
        ensure!(
            selected > 0 && selected <= 2048 && selected < total,
            "GPU BF16 micro fused requires 1..2048 selected positions"
        );
        let (selected, total, heads, kv, dim) = (
            u32::try_from(selected)?,
            u32::try_from(total)?,
            u32::try_from(geometry.query_heads)?,
            u32::try_from(geometry.kv_heads)?,
            u32::try_from(geometry.head_dim)?,
        );
        // SAFETY: Selected indices originate from the certified original
        // CPU micro-region planner and are checked before device upload.
        // In-order OpenCL queue keeps live K/V and index buffers valid.
        unsafe {
            ExecuteKernel::new(&self.kernels.micro_hot_fused)
                .set_arg(query)
                .set_arg(keys)
                .set_arg(values)
                .set_arg(indices)
                .set_arg(output)
                .set_arg(&selected)
                .set_arg(&total)
                .set_arg(&heads)
                .set_arg(&kv)
                .set_arg(&dim)
                .set_global_work_size(geometry.query_heads * LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&self.queue)
                .context("certified micro BF16 GPU-resident fused attention")?;
        }
        Ok(())
    }

    /// Optional exact BF16 GPU QK + softmax + PV in one physical
    /// OpenCL kernel. Unlike sparse micro attention, this computes
    /// EVERY token's real QK and V and has zero omitted probability
    /// mass. The original CPU ColdStore and four-bit index remain
    /// intact; this replaces only the GPU computation of attention.
    pub(crate) fn exact_hot_fused_attention(
        &self,
        query: &Buffer<u8>,
        key: &Buffer<u8>,
        value: &Buffer<u8>,
        output: &Buffer<u8>,
        tokens: usize,
        geometry: AttentionGeometry,
    ) -> Result<()> {
        if let Some(tiled) = &self.tiled_attention {
            ensure!(tokens > 0, "empty tiled decode");
            return tiled.run(self, query, key, value, output, 1, tokens - 1, geometry);
        }
        ensure!(
            (1..=2048).contains(&tokens),
            "fused exact GPU attention supports up to 2048 BF16 KV tokens"
        );
        let (n, q, k, d) = (
            u32::try_from(tokens)?,
            u32::try_from(geometry.query_heads)?,
            u32::try_from(geometry.kv_heads)?,
            u32::try_from(geometry.head_dim)?,
        );
        // SAFETY: All BF16 buffers have original verified shape and
        // valid lengths for n tokens; in-order queue prevents a race
        // with preceding BF16 K/V updates.
        unsafe {
            ExecuteKernel::new(&self.kernels.exact_hot_fused)
                .set_arg(query)
                .set_arg(key)
                .set_arg(value)
                .set_arg(output)
                .set_arg(&n)
                .set_arg(&q)
                .set_arg(&k)
                .set_arg(&d)
                .set_global_work_size(geometry.query_heads * LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&self.queue)
                .context("exact BF16 GPU fused QK-softmax-PV kernel failed")?;
        }
        Ok(())
    }

    /// Refresh GPU-resident certified indices ONLY when the CPU metadata
    /// planner's selected position set actually changes. Repeated
    /// related queries avoid repeated host-to-GPU index uploads.
    pub(crate) fn micro_update_cached_indices(
        &self,
        cache: &mut MicroSelectedDevice,
        positions: &[usize],
        total: usize,
    ) -> Result<()> {
        ensure!(
            !positions.is_empty() && positions.len() <= 2048 && positions.len() <= total,
            "GPU micro hot positions invalid"
        );
        ensure!(
            positions.iter().all(|&pos| pos < total) && positions.windows(2).all(|w| w[0] < w[1]),
            "GPU micro positions must be increasing original token IDs"
        );
        let mapped = positions
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if mapped == cache.last_positions {
            cache.reuse_hits += 1;
            return Ok(());
        }
        // SAFETY: Device buffer is allocated for the largest supported
        // cached micro index set; blocking copy ensures host mapped data
        // remains live during transfer and before fused QK kernel.
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut cache.indices, CL_BLOCKING, 0, &mapped, &[])
                .context("GPU upload changed certified micro selection")?;
        }
        cache.last_positions = mapped;
        cache.upload_calls += 1;
        Ok(())
    }

    /// Read a small BF16 K/V token or Q-vector. Unlike GPU weights, these
    /// are CPU routing metadata and canonical ColdStore payload only.
    pub(crate) fn read_bf16_bytes(&self, source: &Buffer<u8>, count: usize) -> Result<Vec<u8>> {
        let mut result = vec![0u8; count];
        // SAFETY: Source OpenCL BF16 buffer has at least count bytes, and
        // CL_BLOCKING keeps result allocated until copying completes.
        unsafe {
            self.queue
                .enqueue_read_buffer(source, CL_BLOCKING, 0, &mut result, &[])
                .context("GPU canonical LOD BF16 token readback")?;
        }
        Ok(result)
    }

    /// Actual GPU QK + original CPU LOD's 4-bit index/LSH, exact-mass
    /// certificate, canonical verified BF16 ColdStore reads, and GPU PV.
    /// BF16 K stays on GPU. Historical V is NEVER allocated in GPU KV.
    pub(crate) fn attention_gpu_region(
        &self,
        ws: &mut Workspace,
        layer_index: usize,
        tokens: usize,
        lod: &mut GpuRegionStore,
    ) -> Result<()> {
        let geometry = lod.geometry();
        let qheads = u32::try_from(geometry.query_heads)?;
        let kvheads = u32::try_from(geometry.kv_heads)?;
        let dim = u32::try_from(geometry.head_dim)?;
        let tokens_u32 = u32::try_from(tokens)?;
        let stage = ws
            .lod_stage
            .as_mut()
            .context("GPU canonical LOD stage missing")?;
        let keys = &ws.kv[layer_index].key;
        let score_items = tokens
            .checked_mul(geometry.query_heads)
            .context("GPU micro score count overflow")?;
        let micro_selection = if lod.config().epsilon > 0.0 {
            let query =
                self.read_bf16_bytes(&ws.query_rope, geometry.query_elements() * BYTES_PER_BF16)?;
            lod.plan_micro_regions(layer_index, &query)?
        } else {
            None
        };
        let sparse = micro_selection.is_some();
        if let Some(ref chosen) = micro_selection {
            // Sparse execution is authorized ONLY by a positive-epsilon
            // certified upper bound; never by LSH or micro links alone.
            self.micro_sparse_scores(
                &ws.query_rope,
                keys,
                &ws.scores,
                &chosen.positions,
                geometry,
                tokens,
            )?;
        } else {
            // Original GPU QK kernel, exact-score mass certificate and
            // verified ColdStore path remain the fallback whenever
            // region bounds are ambiguous or epsilon is zero.
            // SAFETY: Original OpenCL dense QK signature, original BF16 K.
            unsafe {
                ExecuteKernel::new(&self.kernels.qk)
                    .set_arg(&ws.query_rope)
                    .set_arg(keys)
                    .set_arg(&ws.scores)
                    .set_arg(&tokens_u32)
                    .set_arg(&qheads)
                    .set_arg(&kvheads)
                    .set_arg(&dim)
                    .set_global_work_size(tokens * geometry.query_heads)
                    .enqueue_nd_range(&self.queue)
                    .context("GPU canonical LOD full BF16 QK fallback")?;
            }
        }
        let selection = if let Some(selection) = micro_selection {
            selection
        } else if lod.config().epsilon == 0.0 {
            lod.select_dense_exact(layer_index)?
        } else {
            let mut real_scores = vec![0.0f32; score_items];
            // SAFETY: GPU actual BF16 K scores are read back only
            // when conservative region metadata cannot certify.
            unsafe {
                self.queue
                    .enqueue_read_buffer(&ws.scores, CL_BLOCKING, 0, &mut real_scores, &[])
                    .context("GPU canonical exact-score fallback readback")?;
            }
            let query =
                self.read_bf16_bytes(&ws.query_rope, geometry.query_elements() * BYTES_PER_BF16)?;
            lod.select(layer_index, &query, &real_scores)?
        };
        ensure!(
            !selection.positions.is_empty(),
            "LOD selected no value tokens"
        );

        // SAFETY: Same kernel as dense GPU: scores replaced with softmax
        // probabilities for the ENTIRE exact K history.
        unsafe {
            ExecuteKernel::new(&self.kernels.softmax)
                .set_arg(&ws.scores)
                .set_arg(&tokens_u32)
                .set_arg(&qheads)
                .set_global_work_size(geometry.query_heads)
                .enqueue_nd_range(&self.queue)
                .context("GPU canonical LOD full QK softmax")?;
        }
        let mut masses = vec![1.0f32; geometry.query_heads];
        if selection.omitted > 0 && !sparse {
            // In the fully scored path, renormalize retained V using
            // the true full-softmax mass. In the sparse path, dropped
            // scores were initialized to -INF, and the same softmax
            // already normalizes ONLY the certified retained set.
            let mut probabilities = vec![0.0f32; score_items];
            // SAFETY: All score probabilities produced by GPU are ready,
            // and blocking read stores their exact post-softmax FP32 bits.
            unsafe {
                self.queue
                    .enqueue_read_buffer(&ws.scores, CL_BLOCKING, 0, &mut probabilities, &[])
                    .context("GPU LOD selected mass readback")?;
            }
            for (head, mass) in masses.iter_mut().enumerate() {
                let start = head * tokens;
                let full = probabilities[start..start + tokens]
                    .iter()
                    .map(|&p| p as f64)
                    .sum::<f64>();
                let kept = selection
                    .positions
                    .iter()
                    .map(|&pos| f64::from(probabilities[start + pos]))
                    .sum::<f64>();
                ensure!(
                    full.is_finite() && full > 0.0 && kept > 0.0 && kept.is_finite(),
                    "GPU canonical LOD zero/nonfinite selected softmax mass"
                );
                *mass = (kept / full) as f32;
            }
        }
        let zeros = vec![0.0f32; geometry.query_elements()];
        // SAFETY: Blocking writes initialize accumulator before GPU stages.
        // No V copies are held beyond the configured GPU staging buffer.
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut stage.accumulator, CL_BLOCKING, 0, &zeros, &[])
                .context("initialize GPU canonical LOD PV accumulators")?;
            self.queue
                .enqueue_write_buffer(&mut stage.selected_mass, CL_BLOCKING, 0, &masses, &[])
                .context("upload GPU canonical LOD retained masses")?;
        }
        for chunk in selection.positions.chunks(lod.stage_tokens()) {
            let values = lod.read_values(layer_index, chunk)?;
            let ids = chunk
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            // SAFETY: Staging buffers are sized to stage_tokens and
            // driver copies are blocking and ordered after the previous
            // PV stage. BF16 payload comes from canonical ColdStore.
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut stage.values, CL_BLOCKING, 0, &values, &[])
                    .context("upload verified exact BF16 V from CPU ColdStore")?;
                self.queue
                    .enqueue_write_buffer(&mut stage.indices, CL_BLOCKING, 0, &ids, &[])
                    .context("upload selected exact token coordinates")?;
                ExecuteKernel::new(&self.kernels.lod_pv_stage)
                    .set_arg(&ws.scores)
                    .set_arg(&stage.values)
                    .set_arg(&stage.indices)
                    .set_arg(&stage.accumulator)
                    .set_arg(&u32::try_from(chunk.len())?)
                    .set_arg(&tokens_u32)
                    .set_arg(&qheads)
                    .set_arg(&kvheads)
                    .set_arg(&dim)
                    .set_global_work_size(geometry.query_elements())
                    .enqueue_nd_range(&self.queue)
                    .context("GPU canonical exact BF16 V PV stage")?;
            }
        }
        // SAFETY: All PV stage kernels are in-order on this queue,
        // final kernel converts the aggregate to BF16 once per output.
        unsafe {
            ExecuteKernel::new(&self.kernels.lod_pv_finish)
                .set_arg(&stage.accumulator)
                .set_arg(&stage.selected_mass)
                .set_arg(&ws.attention)
                .set_arg(&qheads)
                .set_arg(&dim)
                .set_global_work_size(geometry.query_elements())
                .enqueue_nd_range(&self.queue)
                .context("GPU canonical LOD final BF16 attention")?;
        }
        Ok(())
    }

    pub(crate) fn argmax(
        &self,
        logits: &Buffer<u8>,
        result: &Buffer<u32>,
        vocab_size: u32,
    ) -> Result<()> {
        // SAFETY: One workgroup reads the logits and writes one token ID.
        unsafe {
            ExecuteKernel::new(&self.kernels.argmax)
                .set_arg(logits)
                .set_arg(result)
                .set_arg(&vocab_size)
                .set_global_work_size(LOCAL_WORK_SIZE)
                .set_local_work_size(LOCAL_WORK_SIZE)
                .enqueue_nd_range(&self.queue)
                .context("GPU BF16 argmax failed")?;
        }
        Ok(())
    }
}

impl GpuWeight {
    pub(crate) fn load(
        engine: &GpuEngine,
        checkpoint: &CheckpointShards,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<Self> {
        checkpoint.with_tensor(name, |view| {
            Self::load_view(engine, name, expected_shape, view)
        })
    }

    pub(crate) fn load_view(
        engine: &GpuEngine,
        name: &str,
        expected_shape: &[usize],
        view: TensorView<'_>,
    ) -> Result<Self> {
        ensure!(view.dtype() == Dtype::BF16, "{name} is not BF16");
        ensure!(
            view.shape() == expected_shape,
            "weight {name} shape mismatch: found {:?}, expected {:?}",
            view.shape(),
            expected_shape
        );
        ensure!(
            view.data().len() <= engine.max_gpu_allocation as usize,
            "weight {name} exceeds max GPU buffer allocation"
        );
        // Only linear projection matrices use native HIP. BF16 token
        // embeddings remain full OpenCL arrays for direct index access,
        // all RMSNorm scales remain OpenCL; no duplicate large weights.
        #[cfg(feature = "hip-backend")]
        if !name.ends_with("embed_tokens.weight")
            && view.shape().len() == 2
            && let Some(hip) = engine.hip.as_ref()
        {
            let hip_weight = hip
                .upload_bf16(view.data())
                .with_context(|| format!("native gfx1103 HIP weight {name}"))?;
            return Ok(Self {
                data: engine.alloc_u8(2)?,
                hip: Some(hip_weight),
            });
        }
        let mut data = engine
            .alloc_u8(view.data().len())
            .with_context(|| format!("GPU memory for weight {name}"))?;
        engine
            .transfer(&mut data, view.data())
            .with_context(|| format!("GPU upload for weight {name}"))?;
        Ok(Self {
            data,
            #[cfg(feature = "hip-backend")]
            hip: None,
        })
    }
}

impl GpuLayer {
    pub(crate) fn load(
        engine: &GpuEngine,
        checkpoint: &CheckpointShards,
        index: usize,
        spec: &ModelSpec,
    ) -> Result<Self> {
        let p = format!("model.layers.{index}");
        let hidden = spec.hidden_size;
        let middle = spec
            .intermediate_size
            .context("model missing intermediate_size")?;
        let qsize = spec.attention.query_elements();
        let kvsize = spec.attention.kv_elements_per_token();
        let weight = |name: &str, shape: &[usize]| {
            GpuWeight::load(engine, checkpoint, &format!("{p}.{name}"), shape)
        };
        Ok(Self {
            input_norm: weight("input_layernorm.weight", &[hidden])?,
            q: weight("self_attn.q_proj.weight", &[qsize, hidden])?,
            k: weight("self_attn.k_proj.weight", &[kvsize, hidden])?,
            v: weight("self_attn.v_proj.weight", &[kvsize, hidden])?,
            o: weight("self_attn.o_proj.weight", &[hidden, qsize])?,
            post_norm: weight("post_attention_layernorm.weight", &[hidden])?,
            gate: weight("mlp.gate_proj.weight", &[middle, hidden])?,
            up: weight("mlp.up_proj.weight", &[middle, hidden])?,
            down: weight("mlp.down_proj.weight", &[hidden, middle])?,
        })
    }
}

impl Workspace {
    pub(crate) fn new(
        engine: &GpuEngine,
        spec: &ModelSpec,
        max_tokens: usize,
        lod: Option<&GpuRegionConfig>,
    ) -> Result<Self> {
        let hidden = spec
            .hidden_size
            .checked_mul(BYTES_PER_BF16)
            .context("hidden bytes overflow")?;
        let qsize = spec
            .attention
            .query_elements()
            .checked_mul(2)
            .context("Q bytes overflow")?;
        let kvsize = spec
            .attention
            .kv_elements_per_token()
            .checked_mul(2)
            .context("KV bytes overflow")?;
        let mid = spec
            .intermediate_size
            .context("missing MLP width")?
            .checked_mul(2)
            .context("MLP bytes overflow")?;
        let vocab = spec
            .vocab_size
            .context("missing vocabulary")?
            .checked_mul(2)
            .context("vocabulary bytes overflow")?;
        let capacity = max_tokens
            .checked_mul(kvsize)
            .context("KV capacity overflow")?;
        let score_count = max_tokens
            .checked_mul(spec.attention.query_heads)
            .context("attention scratch overflow")?;
        let mut kv = Vec::with_capacity(spec.num_hidden_layers);
        for _ in 0..spec.num_hidden_layers {
            kv.push(GpuKv {
                key: engine.alloc_u8(capacity)?,
                value: if lod.is_some() {
                    None
                } else {
                    Some(engine.alloc_u8(capacity)?)
                },
            });
        }
        let lod_stage = if let Some(config) = lod {
            let stage_values = config
                .stage_tokens
                .checked_mul(kvsize)
                .context("GPU BF16 value staging overflow")?;
            Some(LodStage {
                values: engine.alloc_u8(stage_values)?,
                indices: engine.alloc_indices(config.stage_tokens)?,
                accumulator: engine.alloc_scores(spec.attention.query_elements())?,
                selected_mass: engine.alloc_scores(spec.attention.query_heads)?,
            })
        } else {
            None
        };
        Ok(Self {
            hidden0: engine.alloc_u8(hidden)?,
            hidden1: engine.alloc_u8(hidden)?,
            norm1: engine.alloc_u8(hidden)?,
            norm2: engine.alloc_u8(hidden)?,
            query: engine.alloc_u8(qsize)?,
            query_rope: engine.alloc_u8(qsize)?,
            key: engine.alloc_u8(kvsize)?,
            key_rope: engine.alloc_u8(kvsize)?,
            value: engine.alloc_u8(kvsize)?,
            attention: engine.alloc_u8(qsize)?,
            attn_o: engine.alloc_u8(hidden)?,
            gate: engine.alloc_u8(mid)?,
            up: engine.alloc_u8(mid)?,
            activation: engine.alloc_u8(mid)?,
            down: engine.alloc_u8(hidden)?,
            logits: engine.alloc_u8(vocab)?,
            token_id: engine.alloc_result()?,
            scores: engine.alloc_scores(score_count)?,
            kv,
            lod_stage,
        })
    }
}
