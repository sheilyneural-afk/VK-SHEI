//! Native gfx1103 HIP BF16 matrix bridge: opt-in only.
//! Original CPU, OpenCL, BF16 cold store and LOD remain unchanged.
//! No HSA_OVERRIDE_GFX_VERSION: bridge rejects all non-native configurations.
use anyhow::{Context, Result, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

type ErrFn = unsafe extern "C" fn() -> *const c_char;
type NewFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type UploadFn = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> *mut c_void;
type FreeFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type ReadFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut u8, usize) -> i32;
type GemmFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, *const u8, *mut u8, i32, i32, i32) -> i32;

struct Inner {
    _lib: Library,
    handle: NonNull<c_void>,
    last_error: ErrFn,
    destroy: DestroyFn,
    upload: UploadFn,
    free: FreeFn,
    read: ReadFn,
    gemm: GemmFn,
}
// SAFETY: all calls on native HIP handles are serialized by the shared
// Mutex; the bridge additionally uses a C++ mutex. No native handle is
// accessed while either runtime or owned weights are being dropped.
unsafe impl Send for Inner {}

impl Inner {
    fn error(&self, stage: &str, code: i32) -> anyhow::Error {
        // SAFETY: C ABI exposes thread-local null-terminated string.
        let details = unsafe {
            let p = (self.last_error)();
            if p.is_null() {
                String::from("Native HIP error unavailable")
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        anyhow::anyhow!("{stage} failed: code={code}: {details}")
    }
}
impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: Arc shared ownership keeps context alive while any
        // HipWeight still exists.
        unsafe {
            (self.destroy)(self.handle.as_ptr());
        }
    }
}
#[derive(Clone)]
pub struct HipRuntime {
    inner: Arc<Mutex<Inner>>,
}
pub struct HipWeight {
    ptr: NonNull<c_void>,
    runtime: HipRuntime,
    bytes: usize,
}
// SAFETY: pointer refers to immutable HIP BF16 storage; access and freeing
// are only possible through a locked owning HipRuntime.
unsafe impl Send for HipWeight {}
impl Drop for HipWeight {
    fn drop(&mut self) {
        if let Ok(ctx) = self.runtime.inner.lock() {
            // SAFETY: bridge allocates and frees immutable BF16 HIP weights.
            unsafe {
                let _ = (ctx.free)(ctx.handle.as_ptr(), self.ptr.as_ptr());
            }
        }
    }
}
impl HipRuntime {
    pub fn new() -> Result<Self> {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("native/libmslim_hipblaslt_bridge.so");
        ensure!(
            path.is_file(),
            "Native HIP bridge absent at {} (compile for gfx1103)",
            path.display()
        );
        // SAFETY: only exact trusted local library path is loaded.
        let library = unsafe { Library::new(&path) }
            .with_context(|| format!("load native gfx1103 BF16 HIP bridge {}", path.display()))?;
        unsafe fn read_symbol<T: Copy>(lib: &Library, name: &[u8]) -> Result<T> {
            // SAFETY: caller specifies correct extern-C signature from ABI.
            let symbol = unsafe { lib.get::<T>(name) }?;
            Ok(*symbol)
        }
        let (last_error, create, destroy, upload, free, read, gemm) = unsafe {
            (
                read_symbol::<ErrFn>(&library, b"tv_hip_last_error\0")?,
                read_symbol::<NewFn>(&library, b"tv_hip_new\0")?,
                read_symbol::<DestroyFn>(&library, b"tv_hip_destroy\0")?,
                read_symbol::<UploadFn>(&library, b"tv_hip_load_weight\0")?,
                read_symbol::<FreeFn>(&library, b"tv_hip_free_weight\0")?,
                read_symbol::<ReadFn>(&library, b"tv_hip_read_weight\0")?,
                read_symbol::<GemmFn>(&library, b"tv_hip_matrix_multiply\0")?,
            )
        };
        // SAFETY: verifies gfx1103, no architecture overrides or fallback.
        let p = unsafe { create() };
        let handle = NonNull::new(p).ok_or_else(|| {
            // SAFETY: thread-local UTF8 lossy C string.
            let c = unsafe { last_error() };
            let m = if c.is_null() {
                "HIP GPU init returned null".into()
            } else {
                unsafe { CStr::from_ptr(c) }.to_string_lossy().into_owned()
            };
            anyhow::anyhow!("Native HIP BF16 backend unavailable: {m}")
        })?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                _lib: library,
                handle,
                last_error,
                destroy,
                upload,
                free,
                read,
                gemm,
            })),
        })
    }

    pub fn upload_bf16(&self, data: &[u8]) -> Result<HipWeight> {
        ensure!(
            !data.is_empty() && data.len().is_multiple_of(2),
            "HIP BF16 weight must be byte-aligned nonempty 16-bit values"
        );
        let locked = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("native HIP mutex poisoned"))?;
        // SAFETY: original bytes remain allocated during synchronous copy.
        let ptr = unsafe { (locked.upload)(locked.handle.as_ptr(), data.as_ptr(), data.len()) };
        let ptr = NonNull::new(ptr).ok_or_else(|| locked.error("HIP BF16 weight upload", 0))?;
        Ok(HipWeight {
            ptr,
            runtime: self.clone(),
            bytes: data.len(),
        })
    }

    pub fn read_bf16(&self, weight: &HipWeight, offset: usize, output: &mut [u8]) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.inner, &weight.runtime.inner),
            "HIP BF16 weight from different context"
        );
        ensure!(
            offset
                .checked_add(output.len())
                .is_some_and(|n| n <= weight.bytes),
            "HIP weight readback exceeds original buffer"
        );
        let locked = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("HIP mutex poisoned"))?;
        // SAFETY: device pointer allocated by bridge, read range validated.
        let rc = unsafe {
            (locked.read)(
                locked.handle.as_ptr(),
                weight.ptr.as_ptr(),
                offset,
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if rc != 0 {
            return Err(locked.error("native HIP BF16 readback", rc));
        }
        Ok(())
    }

    pub fn gemm_bf16(
        &self,
        weight: &HipWeight,
        input: &[u8],
        output: &mut [u8],
        inner: usize,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.inner, &weight.runtime.inner),
            "HIP weight belongs to a different GPU context"
        );
        ensure!(
            inner > 0 && rows > 0 && (1..=512).contains(&tokens),
            "HIP GEMM dimensions not supported"
        );
        ensure!(
            inner.checked_mul(rows).and_then(|n| n.checked_mul(2)) == Some(weight.bytes),
            "HIP BF16 matrix weight geometry invalid"
        );
        ensure!(
            inner.checked_mul(tokens).and_then(|n| n.checked_mul(2)) == Some(input.len()),
            "HIP matrix BF16 input geometry invalid"
        );
        ensure!(
            rows.checked_mul(tokens).and_then(|n| n.checked_mul(2)) == Some(output.len()),
            "HIP matrix BF16 output geometry invalid"
        );
        let (inner, rows, tokens) = (
            i32::try_from(inner)?,
            i32::try_from(rows)?,
            i32::try_from(tokens)?,
        );
        let locked = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("native GPU mutex poisoned"))?;
        // SAFETY: bridge performs blocking H2D/matmul/D2H under lock.
        let rc = unsafe {
            (locked.gemm)(
                locked.handle.as_ptr(),
                weight.ptr.as_ptr(),
                input.as_ptr(),
                output.as_mut_ptr(),
                inner,
                rows,
                tokens,
            )
        };
        if rc != 0 {
            return Err(locked.error("native gfx1103 hipBLASLt matmul", rc));
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "physical-conformance"))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "physical gfx1103 with locally built HIP bridge required"]
    fn hip_actual_exact_bf16_weight_and_matrix() {
        let device = HipRuntime::new().unwrap();
        let weights = [0x80u8, 0x3f].repeat(256 * 128);
        let w = device.upload_bf16(&weights).unwrap();
        let mut read = vec![0u8; weights.len()];
        for (i, chunk) in read.chunks_mut(4096).enumerate() {
            device.read_bf16(&w, i * 4096, chunk).unwrap();
        }
        assert_eq!(weights, read);
        let input = [0x80u8, 0x3f].repeat(128 * 16);
        let mut output = vec![0u8; 256 * 16 * 2];
        device
            .gemm_bf16(&w, &input, &mut output, 128, 256, 16)
            .unwrap();
        assert!(output.chunks_exact(2).all(|x| x == [0, 0x43]));
    }
}
