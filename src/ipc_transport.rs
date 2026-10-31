//! Length-delimited local IPC transport for the generic runtime protocol.
use crate::generic_runtime::{
    IpcRequest, IpcResponse, decode_ipc_request, decode_ipc_response, encode_ipc,
    encode_ipc_response,
};
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

pub struct IpcClient {
    stream: UnixStream,
}
impl IpcClient {
    pub fn connect(path: impl AsRef<Path>, timeout: Duration) -> Result<Self> {
        let stream = UnixStream::connect(path).context("connect generic runtime IPC")?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        Ok(Self { stream })
    }
    pub fn request(&mut self, request: &IpcRequest) -> Result<IpcResponse> {
        let body = encode_ipc(request)?;
        write_frame(&mut self.stream, &body)?;
        let response = read_frame(&mut self.stream)?;
        decode_ipc_response(&response)
    }
}

pub fn write_request(stream: &mut UnixStream, request: &IpcRequest) -> Result<()> {
    write_frame(stream, &encode_ipc(request)?)
}
pub fn read_request(stream: &mut UnixStream) -> Result<IpcRequest> {
    decode_ipc_request(&read_frame(stream)?)
}
pub fn write_response(stream: &mut UnixStream, response: &IpcResponse) -> Result<()> {
    write_frame(stream, &encode_ipc_response(response)?)
}
pub fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>> {
    let mut header = [0u8; 4];
    stream
        .read_exact(&mut header)
        .context("read IPC frame header")?;
    let len = u32::from_be_bytes(header) as usize;
    if len == 0 || len > MAX_FRAME_BYTES {
        bail!("invalid IPC frame length");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).context("read IPC frame")?;
    Ok(body)
}
pub fn write_frame(stream: &mut UnixStream, body: &[u8]) -> Result<()> {
    if body.is_empty() || body.len() > MAX_FRAME_BYTES {
        bail!("invalid IPC frame body");
    }
    let len = u32::try_from(body.len()).context("IPC frame too large")?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}
