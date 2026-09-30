/// Common neural-network activation functions.

/// Sigmoid activation.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU activation.
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Hyperbolic tangent.
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Leaky ReLU with a small negative slope.
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

/// Softmax over a slice (stable implementation).
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sigmoid_zero() { assert!((sigmoid(0.0) - 0.5).abs() < 1e-9); }
    #[test]
    fn test_relu() { assert_eq!(relu(-1.0), 0.0); assert_eq!(relu(2.0), 2.0); }
    #[test]
    fn test_softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}

/// FNV-1a 64 — the digest every substrate in the SuperInstance fleet agrees on.
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// True if this crate's FNV-1a still agrees with the rest of the fleet.
pub fn canary_holds() -> bool {
    fnv1a64("café Δ 日本語".as_bytes()) == 0x024a555471370b18d
}
