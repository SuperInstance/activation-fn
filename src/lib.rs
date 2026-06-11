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
