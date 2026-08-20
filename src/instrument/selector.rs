//! Constant-cost capture selection for keeping instrumentation off the hot path.

/// Select exactly one one-based workload invocation for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureSelector {
    selected_invocation: u64,
}

impl CaptureSelector {
    pub fn new(selected_invocation: u64) -> anyhow::Result<Self> {
        anyhow::ensure!(
            selected_invocation > 0,
            "selected capture invocation must be one-based"
        );
        Ok(Self {
            selected_invocation,
        })
    }

    #[inline]
    pub fn is_selected(self, invocation: u64) -> bool {
        invocation == self.selected_invocation
    }

    pub fn selected_invocation(self) -> u64 {
        self.selected_invocation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_only_the_configured_one_based_invocation() {
        let selector = CaptureSelector::new(3).unwrap();
        assert!(!selector.is_selected(2));
        assert!(selector.is_selected(3));
        assert!(!selector.is_selected(4));
        assert!(CaptureSelector::new(0).is_err());
    }
}
