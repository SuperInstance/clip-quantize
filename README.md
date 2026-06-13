# Clip-Quantize — Range Clipping and Uniform Quantization

**Clip-quantize** is a two-step signal processing operation: first **clip** (clamp) continuous values to a bounded range, then **quantize** them into a fixed number of discrete bins. This is the fundamental operation behind analog-to-digital conversion, color depth reduction, histogram computation, and neural network weight quantization.

## Why It Matters

Every time you convert a floating-point model to INT8 for edge deployment, downsample a 16-bit audio recording to 8-bit, or bucketize features for a histogram, you're clip-quantizing. The operation loses information — the art is in choosing the range and bin count that minimize perceptual or predictive loss. Quantization-aware training (QAT) and post-training quantization (PTQ) both depend on exactly this operation. In ML compilers like Apache TVM and in ONNX Runtime, `clip + quantize` is a fused kernel that runs billions of times per inference batch.

## How It Works

Given an input value `x`, a range `[lo, hi]`, and `B` bins:

**Step 1 — Clip** (clamp to range):

```
x_clamped = max(lo, min(hi, x))
```

**Step 2 — Normalize** to `[0, 1)`:

```
x_norm = (x_clamped - lo) / (hi - lo)
```

**Step 3 — Quantize** (map to bin index):

```
bin = floor(x_norm × B)
bin = min(bin, B - 1)   // clamp upper boundary
```

**Dequantization** (inverse for reconstruction):

```
x_reconstructed = lo + (bin + 0.5) × (hi - lo) / B
```

The `+0.5` places the reconstructed value at the center of each bin, minimizing worst-case reconstruction error to `Δ/2` where `Δ = (hi - lo) / B`.

**Quantization error**: For uniform quantization, the mean squared error (MSE) is:

```
MSE = Δ² / 12
```

This is the classical Bennett integral result for uniform quantization under a smooth input distribution. The signal-to-quantization-noise ratio (SQNR) increases by ~6 dB per additional bit.

**Complexity**: `O(1)` per value, `O(n)` for a slice of `n` values. The `quantize_slice` function operates in-place, replacing each value with its dequantized reconstruction while returning the bin indices.

## Quick Start

```rust
fn clip_quantize(value: f64, lo: f64, hi: f64, bins: usize) -> usize {
    let clamped = value.clamp(lo, hi);
    let normalized = (clamped - lo) / (hi - lo);
    ((normalized * bins as f64).floor() as usize).min(bins - 1)
}

fn main() {
    let mut data = vec![-1.5, 0.2, 0.7, 1.1, 2.8, -0.3, 0.5, 1.9];

    // Clip to [-1.0, 2.0] and quantize into 4 bins
    for v in &data {
        let bin = clip_quantize(*v, -1.0, 2.0, 4);
        println!("Value {:>5.1} → bin {}", v, bin);
    }
    // -1.5 → bin 0 (clipped to -1.0)
    //  2.8 → bin 3 (clipped to  2.0)
    //  0.7 → bin 2
}
```

## API

| Function | Description |
|---|---|
| `clip_quantize(value, range, bins)` | Clip a single value to `range` and return its bin index. `O(1)`. |
| `quantize_slice(data, range, bins)` | In-place: quantizes each element, replaces with dequantized center, returns bin indices. `O(n)`. |

## Architecture Notes

Clip-quantize is a signal-processing primitive in the γ (generation) side of γ + η = C — it transforms continuous representations into discrete ones for efficient storage, transmission, and compute. In SuperInstance, it's used for metric bucketization and model compression. See [SuperInstance Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Gray, R. M. & Neuhoff, D. L. (1998). *Quantization*. IEEE Transactions on Information Theory 44(6), 2325–2383. — Definitive survey of quantization theory.
2. Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR. — How clip+quantize works in ML quantization.
3. Gersho, A. & Gray, R. M. (1991). *Vector Quantization and Signal Compression*. Kluwer Academic.

## License

MIT
