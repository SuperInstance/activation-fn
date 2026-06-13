# Activation Functions

**Activation functions** introduce non-linearity into neural networks, determining whether a neuron should fire based on its weighted inputs.

## Why It Matters

Without activation functions, deep networks collapse into linear transformations regardless of depth. The choice of activation affects training speed, gradient flow, and representational power. Modern activations (ReLU, GELU, SwiGLU) are critical to LLM performance.

## How It Works

Implements common activation functions: sigmoid (σ(x) = 1/(1+e⁻ˣ)), tanh, ReLU (max(0,x)), Leaky ReLU, ELU, GELU, and SwiGLU. Each includes forward and derivative implementations for backpropagation.

## Usage

```toml
[dependencies]
activation-fn = "0.1.0"
```

```rust
use activation_fn;

// See examples/ directory for detailed usage
```

## API

- `sigmoid` (lib.rs)
- `relu` (lib.rs)
- `tanh` (lib.rs)
- `leaky_relu` (lib.rs)
- `softmax` (lib.rs)

## Architecture

This crate is part of the **[SuperInstance](https://github.com/SuperInstance)** ecosystem — a conservation-law-based framework for fleet coordination, ternary computation, and distributed agent systems.

### Related Crates

- [`superinstance-core`](https://github.com/SuperInstance/superinstance-core) — Core conservation law (γ + η = C)
- [`superinstance-harness`](https://github.com/SuperInstance/superinstance-harness) — Build harness and self-improving loop
- [`fleet-coordinator`](https://github.com/SuperInstance/fleet-coordinator) — Fleet-level coordination

## References

- [SuperInstance Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)
- [Conservation Law Paper](https://github.com/SuperInstance/SuperInstance/blob/main/docs/conservation-law.md)

## License

MIT
