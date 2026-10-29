//! Hardware OpenCL backend for the portable Rust KV attention reference.
//! Executes QK, stable softmax and PV kernels on a CL_DEVICE_TYPE_GPU.
//! Does NOT implement LLM embedding, RMSNorm, RoPE, MLP or weight inference.
//! All BF16/F16/F32 input payloads are transferred without quantization.
//! GPU computes in FP32 and is compared against the Rust CPU f64 reference.

use crate::portable::{AttentionBackend, ExactKvFrames, FloatFormat, PortableError};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use serde::Serialize;
use std::{ptr, sync::Mutex, time::Instant};

fn driver_error(stage: &str, error: impl std::fmt::Display) -> PortableError {
    PortableError::BackendFailure(format!("OpenCL {stage}: {error}"))
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub vendor_id: u32,
    pub driver_version: String,
    pub opencl_device_type: u64,
    pub max_compute_units: u32,
    pub reported_global_memory_bytes: u64,
    pub max_allocation_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct GpuTimings {
    /// Total synchronous host-to-GPU + QK + softmax + PV + readback time.
    pub dispatch_ms: f64,
    pub qk_kernel_ms: f64,
    pub softmax_kernel_ms: f64,
    pub pv_kernel_ms: f64,
    pub input_kv_bytes: usize,
    pub scratch_score_bytes: usize,
    pub output_bytes: usize,
}

struct Runtime {
    context: Context,
    program: Program,
    queue: CommandQueue,
    last_timings: Option<GpuTimings>,
}

pub struct OpenClBackend {
    info: GpuDeviceInfo,
    runtime: Mutex<Runtime>,
}

impl OpenClBackend {
    pub fn new() -> Result<Self, PortableError> {
        let all =
            get_all_devices(CL_DEVICE_TYPE_GPU).map_err(|e| driver_error("enumerate GPU", e))?;
        let requested = std::env::var("TERRSVUSION_OPENCL_DEVICE")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let device = all
            .into_iter()
            .map(Device::new)
            .find(|dev| {
                let n = dev.name().unwrap_or_default().to_ascii_lowercase();
                let v = dev.vendor().unwrap_or_default().to_ascii_lowercase();
                requested.is_empty() || n.contains(&requested) || v.contains(&requested)
            })
            .ok_or_else(|| {
                PortableError::UnsupportedDevice(format!(
                    "no OpenCL GPU matches device filter {requested:?}"
                ))
            })?;
        let device_type = device
            .dev_type()
            .map_err(|e| driver_error("device type", e))?;
        if device_type & CL_DEVICE_TYPE_GPU == 0 {
            return Err(PortableError::UnsupportedDevice(
                "OpenCL selected device is not a GPU".into(),
            ));
        }
        // OpenCL kernel loads aligned LE u16/u32 directly from raw bytes.
        // Reject incompatible big-endian devices instead of silently
        // interpreting BF16/F16/F32 backwards.
        if !device
            .endian_little()
            .map_err(|e| driver_error("device endian", e))?
        {
            return Err(PortableError::UnsupportedDevice(
                "GPU is big-endian; LE fast path cannot be used".into(),
            ));
        }
        let info = GpuDeviceInfo {
            name: device.name().map_err(|e| driver_error("device name", e))?,
            vendor: device.vendor().map_err(|e| driver_error("vendor", e))?,
            vendor_id: device
                .vendor_id()
                .map_err(|e| driver_error("vendor ID", e))?,
            driver_version: device
                .driver_version()
                .map_err(|e| driver_error("driver", e))?,
            opencl_device_type: device_type,
            max_compute_units: device
                .max_compute_units()
                .map_err(|e| driver_error("compute units", e))?,
            reported_global_memory_bytes: device
                .global_mem_size()
                .map_err(|e| driver_error("global memory", e))?,
            max_allocation_bytes: device
                .max_mem_alloc_size()
                .map_err(|e| driver_error("max allocation", e))?,
        };
        let context =
            Context::from_device(&device).map_err(|e| driver_error("create GPU context", e))?;
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| driver_error("create GPU command queue", e))?;
        let program = Program::create_and_build_from_source(
            &context,
            include_str!("../assets/opencl_attention.cl"),
            "-cl-std=CL1.2",
        )
        .map_err(|e| driver_error("compile GPU attention kernels", e))?;
        Ok(Self {
            info,
            runtime: Mutex::new(Runtime {
                context,
                program,
                queue,
                last_timings: None,
            }),
        })
    }

    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    pub fn last_timings(&self) -> Option<GpuTimings> {
        self.runtime
            .lock()
            .ok()
            .and_then(|runtime| runtime.last_timings)
    }

    fn dispatch(&self, cache: &ExactKvFrames, query: &[f32]) -> Result<Vec<f32>, PortableError> {
        let geometry = cache.geometry();
        if query.len() != geometry.query_elements() {
            return Err(PortableError::ShapeMismatch("OpenCL Q shape"));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(PortableError::NonFinite("OpenCL Q"));
        }
        if cache.is_empty() {
            return Ok(vec![0.0; query.len()]);
        }
        let start = Instant::now();
        let tokens = u32::try_from(cache.tokens())
            .map_err(|_| PortableError::InvalidGeometry("OpenCL token count > u32"))?;
        let qheads = u32::try_from(geometry.query_heads)
            .map_err(|_| PortableError::InvalidGeometry("OpenCL qheads > u32"))?;
        let kvheads = u32::try_from(geometry.kv_heads)
            .map_err(|_| PortableError::InvalidGeometry("OpenCL kvheads > u32"))?;
        let dimension = u32::try_from(geometry.head_dim)
            .map_err(|_| PortableError::InvalidGeometry("OpenCL head_dim > u32"))?;
        let score_items = (tokens as usize).checked_mul(geometry.query_heads).ok_or(
            PortableError::InvalidGeometry("OpenCL score indexing overflow"),
        )?;
        // Kernels use 32-bit unsigned global indices.
        u32::try_from(score_items)
            .map_err(|_| PortableError::InvalidGeometry("OpenCL score count exceeds u32 range"))?;
        u32::try_from(geometry.query_elements())
            .map_err(|_| PortableError::InvalidGeometry("OpenCL output count exceeds u32 range"))?;
        let score_size = score_items
            .checked_mul(4)
            .ok_or(PortableError::InvalidGeometry(
                "OpenCL score allocation overflow",
            ))?;
        if score_size as u64 > self.info.max_allocation_bytes {
            return Err(PortableError::InvalidGeometry(
                "OpenCL score allocation exceeds device limit",
            ));
        }
        let (first_key, first_value) = &cache.frames()[0];
        let key_dtype = first_key.dtype();
        let value_dtype = first_value.dtype();
        let mut kraw = Vec::new();
        let mut vraw = Vec::new();
        for (key, value) in cache.frames() {
            if key_dtype != key.dtype() || value_dtype != value.dtype() {
                return Err(PortableError::BackendFailure(
                    "OpenCL backend requires uniform K and V formats across frames".into(),
                ));
            }
            kraw.extend_from_slice(key.as_le_bytes());
            vraw.extend_from_slice(value.as_le_bytes());
        }
        if kraw.len() as u64 > self.info.max_allocation_bytes
            || vraw.len() as u64 > self.info.max_allocation_bytes
        {
            return Err(PortableError::InvalidGeometry(
                "OpenCL KV allocation exceeds device limit",
            ));
        }

        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| PortableError::BackendFailure("GPU queue mutex poisoned".into()))?;
        // SAFETY: Valid and bounded lengths, owned buffers and live OpenCL context.
        let mut q_buf = unsafe {
            Buffer::<f32>::create(
                &runtime.context,
                CL_MEM_READ_ONLY,
                query.len(),
                ptr::null_mut(),
            )
            .map_err(|e| driver_error("allocate Q", e))?
        };
        let mut k_buf = unsafe {
            Buffer::<u8>::create(
                &runtime.context,
                CL_MEM_READ_ONLY,
                kraw.len(),
                ptr::null_mut(),
            )
            .map_err(|e| driver_error("allocate K", e))?
        };
        let mut v_buf = unsafe {
            Buffer::<u8>::create(
                &runtime.context,
                CL_MEM_READ_ONLY,
                vraw.len(),
                ptr::null_mut(),
            )
            .map_err(|e| driver_error("allocate V", e))?
        };
        let scores = unsafe {
            Buffer::<f32>::create(
                &runtime.context,
                CL_MEM_READ_WRITE,
                score_size / 4,
                ptr::null_mut(),
            )
            .map_err(|e| driver_error("allocate score scratch", e))?
        };
        let output = unsafe {
            Buffer::<f32>::create(
                &runtime.context,
                CL_MEM_WRITE_ONLY,
                query.len(),
                ptr::null_mut(),
            )
            .map_err(|e| driver_error("allocate output", e))?
        };
        // SAFETY: blocking transfers hold host bytes alive until DMA completion.
        unsafe {
            runtime
                .queue
                .enqueue_write_buffer(&mut q_buf, CL_BLOCKING, 0, query, &[])
                .map_err(|e| driver_error("write Q", e))?;
            runtime
                .queue
                .enqueue_write_buffer(&mut k_buf, CL_BLOCKING, 0, &kraw, &[])
                .map_err(|e| driver_error("write K", e))?;
            runtime
                .queue
                .enqueue_write_buffer(&mut v_buf, CL_BLOCKING, 0, &vraw, &[])
                .map_err(|e| driver_error("write V", e))?;
        }
        let dtype_code = |dtype| match dtype {
            FloatFormat::Bf16 => 1u32,
            FloatFormat::F16 => 2u32,
            FloatFormat::F32 => 3u32,
        };
        let k_qk = Kernel::create(&runtime.program, "terrsvusion_qk")
            .map_err(|e| driver_error("QK kernel", e))?;
        let k_softmax = Kernel::create(&runtime.program, "terrsvusion_softmax")
            .map_err(|e| driver_error("softmax kernel", e))?;
        let k_pv = Kernel::create(&runtime.program, "terrsvusion_pv")
            .map_err(|e| driver_error("PV kernel", e))?;

        // SAFETY: OpenCL command queue is serialized by mutex, kernel
        // signature exactly matches buffers and eight/small scalar arguments.
        let qk_event = unsafe {
            ExecuteKernel::new(&k_qk)
                .set_arg(&q_buf)
                .set_arg(&k_buf)
                .set_arg(&scores)
                .set_arg(&tokens)
                .set_arg(&qheads)
                .set_arg(&kvheads)
                .set_arg(&dimension)
                .set_arg(&dtype_code(key_dtype))
                .set_global_work_size((tokens as usize) * geometry.query_heads)
                .enqueue_nd_range(&runtime.queue)
                .map_err(|e| driver_error("execute QK", e))?
        };
        let softmax_event = unsafe {
            ExecuteKernel::new(&k_softmax)
                .set_arg(&scores)
                .set_arg(&tokens)
                .set_arg(&qheads)
                .set_global_work_size(geometry.query_heads)
                .enqueue_nd_range(&runtime.queue)
                .map_err(|e| driver_error("execute softmax", e))?
        };
        let pv_event = unsafe {
            ExecuteKernel::new(&k_pv)
                .set_arg(&scores)
                .set_arg(&v_buf)
                .set_arg(&output)
                .set_arg(&tokens)
                .set_arg(&qheads)
                .set_arg(&kvheads)
                .set_arg(&dimension)
                .set_arg(&dtype_code(value_dtype))
                .set_global_work_size(geometry.query_elements())
                .enqueue_nd_range(&runtime.queue)
                .map_err(|e| driver_error("execute PV", e))?
        };
        let mut output_values = vec![0f32; query.len()];
        // SAFETY: readback is blocking and buffer remains live.
        unsafe {
            runtime
                .queue
                .enqueue_read_buffer(&output, CL_BLOCKING, 0, &mut output_values, &[])
                .map_err(|e| driver_error("GPU output readback", e))?;
        }
        runtime
            .queue
            .finish()
            .map_err(|e| driver_error("finish GPU", e))?;
        let event_ms = |ev: &opencl3::event::Event| -> f64 {
            let timestamps = (ev.profiling_command_start(), ev.profiling_command_end());
            match timestamps {
                (Ok(start), Ok(end)) if end >= start => (end - start) as f64 / 1e6,
                _ => 0.0,
            }
        };
        runtime.last_timings = Some(GpuTimings {
            dispatch_ms: start.elapsed().as_secs_f64() * 1000.0,
            qk_kernel_ms: event_ms(&qk_event),
            softmax_kernel_ms: event_ms(&softmax_event),
            pv_kernel_ms: event_ms(&pv_event),
            input_kv_bytes: kraw.len() + vraw.len(),
            scratch_score_bytes: score_size,
            output_bytes: query.len() * 4,
        });
        if output_values.iter().any(|value| !value.is_finite()) {
            return Err(PortableError::NonFinite("OpenCL output"));
        }
        Ok(output_values)
    }
}

impl AttentionBackend for OpenClBackend {
    fn name(&self) -> &'static str {
        "rust-opencl-gpu-qk-softmax-pv"
    }
    fn decode(&self, cache: &ExactKvFrames, query: &[f32]) -> Result<Vec<f32>, PortableError> {
        self.dispatch(cache, query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portable::{AttentionGeometry, CpuReferenceBackend, RawTensor};

    fn gpu_roundtrip(dtype: FloatFormat) {
        let device = OpenClBackend::new().expect("physical GPU required");
        assert!(device.device_info().opencl_device_type & CL_DEVICE_TYPE_GPU != 0);
        let shape = AttentionGeometry::new(4, 2, 4).unwrap();
        let k: Vec<f32> = (0..32).map(|i| (i as f32 % 11.0 - 5.0) * 0.0625).collect();
        let v: Vec<f32> = (0..32).map(|i| (i as f32 % 13.0 - 6.0) * 0.0625).collect();
        let raw = |a: &[f32]| -> Vec<u8> {
            a.iter()
                .flat_map(|&f| match dtype {
                    FloatFormat::Bf16 => half::bf16::from_f32(f).to_bits().to_le_bytes().to_vec(),
                    FloatFormat::F16 => half::f16::from_f32(f).to_bits().to_le_bytes().to_vec(),
                    FloatFormat::F32 => f.to_le_bytes().to_vec(),
                })
                .collect()
        };
        let mut cache = ExactKvFrames::new(shape);
        cache
            .append(
                RawTensor::from_le_bytes(dtype, vec![4, 2, 4], raw(&k)).unwrap(),
                RawTensor::from_le_bytes(dtype, vec![4, 2, 4], raw(&v)).unwrap(),
            )
            .unwrap();
        let query = vec![0.15; 16];
        let cpu = cache.decode(&CpuReferenceBackend, &query).unwrap();
        let gpu = cache.decode(&device, &query).unwrap();
        for (&a, &b) in cpu.iter().zip(&gpu) {
            assert!((a - b).abs() < 0.0001, "cpu={a}, gpu={b}");
        }
        assert!(device.last_timings().unwrap().qk_kernel_ms > 0.0);
    }

    fn gpu_geometry_roundtrip(query_heads: usize, kv_heads: usize) {
        let backend = OpenClBackend::new().expect("physical GPU required");
        let geometry = AttentionGeometry::new(query_heads, kv_heads, 8).unwrap();
        let tokens = 5usize;
        let count = tokens * kv_heads * 8;
        let key: Vec<f32> = (0..count)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.0625)
            .collect();
        let value: Vec<f32> = (0..count)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.0625)
            .collect();
        let mut cache = ExactKvFrames::new(geometry);
        cache
            .append(
                RawTensor::from_f32(vec![tokens, kv_heads, 8], &key).unwrap(),
                RawTensor::from_f32(vec![tokens, kv_heads, 8], &value).unwrap(),
            )
            .unwrap();
        let query: Vec<f32> = (0..geometry.query_elements())
            .map(|i| ((i % 7) as f32 - 3.0) * 0.125)
            .collect();
        let cpu = cache.decode(&CpuReferenceBackend, &query).unwrap();
        let gpu = cache.decode(&backend, &query).unwrap();
        for (&c, &g) in cpu.iter().zip(gpu.iter()) {
            assert!((c - g).abs() < 0.0001, "cpu={c} gpu={g}");
        }
    }

    #[test]
    #[ignore = "physical GPU required; invoke explicitly with --ignored"]
    fn gpu_actual_mha_4q_4kv() {
        gpu_geometry_roundtrip(4, 4);
    }

    #[test]
    #[ignore = "physical GPU required; invoke explicitly with --ignored"]
    fn gpu_actual_mqa_4q_1kv() {
        gpu_geometry_roundtrip(4, 1);
    }

    #[test]
    #[ignore = "physical GPU required; invoke explicitly with --ignored"]
    fn gpu_actual_bf16() {
        gpu_roundtrip(FloatFormat::Bf16);
    }
    #[test]
    #[ignore = "physical GPU required; invoke explicitly with --ignored"]
    fn gpu_actual_f16() {
        gpu_roundtrip(FloatFormat::F16);
    }
    #[test]
    #[ignore = "physical GPU required; invoke explicitly with --ignored"]
    fn gpu_actual_f32() {
        gpu_roundtrip(FloatFormat::F32);
    }
}
