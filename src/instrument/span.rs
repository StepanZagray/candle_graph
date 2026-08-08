//! RAII span guards for [`super::session::TraceSession`].

use std::time::Instant;

pub use crate::trace::schema::SpanKind;

/// Opaque numeric span identity; formatted only when emitting JSONL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpanId(pub(crate) u64);

/// Fields for one timed op observation inside an open span.
#[derive(Debug, Clone, Copy)]
pub struct OpRecord<'a> {
    pub op_name: &'a str,
    pub inputs: &'a [String],
    pub output: Option<&'a str>,
    pub shape: &'a [usize],
    pub dtype: &'a str,
    pub device: &'a str,
    pub duration_ns: u64,
    pub timestamp_ns: u64,
    pub storage_bytes: Option<u64>,
    pub input_storage_bytes: u64,
    pub category: Option<crate::trace::MemoryCategory>,
}

/// Fields for an explicit memory alloc/free event.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRecord<'a> {
    pub tensor_id: &'a str,
    pub device: &'a str,
    pub bytes: u64,
    pub dtype: &'a str,
    pub category: crate::trace::MemoryCategory,
    pub timestamp_ns: Option<u64>,
    pub op_name: Option<&'a str>,
    pub shape: &'a [usize],
}

/// Fields for a tensor memory observation inside an open span.
#[derive(Debug, Clone, Copy)]
pub struct TensorRecord<'a> {
    pub tensor_id: &'a str,
    pub shape: &'a [usize],
    pub dtype: &'a str,
    pub device: &'a str,
    pub requires_grad: bool,
    pub storage_bytes: Option<u64>,
    pub category: crate::trace::MemoryCategory,
}

impl SpanId {
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl SpanGuard<'_> {
    pub fn id(&self) -> SpanId {
        self.id
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
}

/// RAII span: emits `span_start` on creation and `span_end` with wall duration on drop.
pub struct SpanGuard<'a> {
    pub(crate) session: &'a super::session::TraceSession,
    pub(crate) id: SpanId,
    pub(crate) started: Instant,
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        let duration_ns = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let _ = self.session.end_span(self.id, duration_ns);
    }
}
