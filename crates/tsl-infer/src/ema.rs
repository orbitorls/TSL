//! Exponential moving average over probability vectors (port of `EMABuffer`).

/// Smooths per-frame class probabilities.
#[derive(Debug, Clone)]
pub struct EmaBuffer {
    alpha: f32,
    smoothed: Option<Vec<f32>>,
}

impl EmaBuffer {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            smoothed: None,
        }
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn reset(&mut self) {
        self.smoothed = None;
    }

    /// Update with a 1-D probability vector; returns the smoothed copy.
    pub fn update(&mut self, probs: &[f32]) -> Result<Vec<f32>, EmaError> {
        if probs.is_empty() {
            return Err(EmaError::Empty);
        }
        match &mut self.smoothed {
            None => {
                self.smoothed = Some(probs.to_vec());
            }
            Some(prev) => {
                if prev.len() != probs.len() {
                    *prev = probs.to_vec();
                } else {
                    let a = self.alpha;
                    let inv = 1.0 - a;
                    for (p, x) in prev.iter_mut().zip(probs.iter()) {
                        *p = a * *x + inv * *p;
                    }
                }
            }
        }
        Ok(self.smoothed.as_ref().unwrap().clone())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmaError {
    #[error("probability vector must not be empty")]
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_update_copies() {
        let mut ema = EmaBuffer::new(0.4);
        let out = ema.update(&[0.2, 0.8]).unwrap();
        assert_eq!(out, vec![0.2, 0.8]);
    }

    #[test]
    fn second_update_blends() {
        let mut ema = EmaBuffer::new(0.5);
        ema.update(&[1.0, 0.0]).unwrap();
        let out = ema.update(&[0.0, 1.0]).unwrap();
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }
}
