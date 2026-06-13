# Activation Functions

**Activation Functions** is a Rust library implementing the core mathematical functions used in neural networks — sigmoid, ReLU, tanh, leaky ReLU, and softmax — with numerically stable formulations suitable for both forward inference and gradient computation.

## Why It Matters

Activation functions introduce non-linearity into neural networks, enabling them to approximate arbitrary continuous functions (Universal Approximation Theorem). Without non-linear activations, a deep network reduces to a single linear transform: stacking linear layers yields `f(x) = W₂(W₁x) = (W₂W₁)x`, collapsing all depth into one matrix. The choice of activation function directly impacts training dynamics: vanishing gradients (sigmoid), dying neurons (ReLU), and computational cost (softmax in large vocabularies) are all activation-dependent challenges. This crate provides reference implementations with numerical stability safeguards, serving as both an educational resource and a building block for from-scratch network implementations.

## How It Works

Each activation function maps a real-valued input to an output with specific mathematical properties:

**Sigmoid:** σ(x) = 1 / (1 + e^(−x)). Output range (0, 1). Historically dominant, now mostly used in binary classification output layers. Suffers vanishing gradients: σ'(x) = σ(x)(1 − σ(x)) has maximum value 0.25 at x = 0, meaning gradients shrink exponentially in deep networks. The implementation uses `(-x).exp()` directly.

**ReLU (Rectified Linear Unit):** f(x) = max(0, x). Computationally trivial (single comparison). Gradient is either 0 or 1, eliminating the vanishing gradient problem for positive inputs. However, negative inputs produce zero gradient permanently — the "dying ReLU" problem. Complexity: O(1).

**Hyperbolic Tangent:** tanh(x) = (e^x − e^(−x)) / (e^x + e^(−x)). Output range (−1, 1). Zero-centered, which helps gradient descent converge faster than sigmoid. Still saturates for large |x|.

**Leaky ReLU:** f(x) = x if x > 0, else αx (typically α = 0.01). Prevents dying neurons by maintaining a small gradient (α) for negative inputs. The slope α is a hyperparameter — when learned per-neuron, it becomes Parametric ReLU (PReLU).

**Softmax (numerically stable):**
```
softmax(xᵢ) = e^(xᵢ − max(x)) / Σⱼ e^(xⱼ − max(x))
```
Subtracting `max(x)` before exponentiation prevents overflow: without it, `e^1000 = ∞` in floating point. This is mathematically equivalent because the max cancels in the normalized ratio. Complexity: O(n) for n logits.

| Function | Output Range | Gradient at 0 | Cost |
|----------|-------------|---------------|------|
| Sigmoid | (0, 1) | 0.25 | exp call |
| ReLU | [0, ∞) | 1 (right) | 1 comparison |
| tanh | (−1, 1) | 1.0 | 2 exp calls |
| Leaky ReLU | (−∞, ∞) | 1 or α | 1 comparison |
| Softmax | (0, 1), Σ=1 | varies | O(n) |

## Quick Start

```rust
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
fn relu(x: f64) -> f64 { x.max(0.0) }
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn main() {
    println!("sigmoid(0) = {}", sigmoid(0.0));  // 0.5
    println!("relu(-3) = {}", relu(-3.0));       // 0.0
    let probs = softmax(&[1.0, 2.0, 3.0]);
    println!("softmax = {:?}, sum = {:.4}", probs, probs.iter().sum::<f64>());
}
```

## API

| Function | Signature | Notes |
|----------|-----------|-------|
| `sigmoid` | `(f64) → f64` | Output ∈ (0, 1) |
| `relu` | `(f64) → f64` | max(0, x) |
| `tanh` | `(f64) → f64` | Wrapper around `f64::tanh` |
| `leaky_relu` | `(f64, alpha: f64) → f64` | Negative slope parameterized |
| `softmax` | `(&[f64]) → Vec<f64>` | Numerically stable, O(n) |

## Architecture Notes

Activation functions are the **non-linearity injection points** in the SuperInstance neural architecture. Within the γ + η = C conservation framework, activation choice controls the flow of gradient information between the γ (physical/action) and η (model/prediction) layers. ReLU and its variants enable deep networks to learn the avoidance-dominant behavior patterns documented in the dissertation's Law 2 (294:1 avoid-to-choose ratio).

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Cybenko, G. (1989). "Approximation by Superpositions of a Sigmoidal Function." *Mathematics of Control, Signals, and Systems*, 2(4), 303–314.
2. Nair, V. & Hinton, G. (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines." *ICML*.
3. Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 6: Deep Feedforward Networks.

## License

MIT
