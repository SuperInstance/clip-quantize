# Clip-Quantize

**Range clipping and uniform quantization** for signal processing, ML model compression, and analog-to-digital conversion modeling.

## Why It Matters

Every time you convert a floating-point neural network to INT8 for edge deployment, downsample a 16-bit audio recording to 8-bit, or bucketize features for a histogram, you're clip-quantizing. The operation loses information — the art is in choosing the range and bin count that minimize perceptual or predictive loss.

**Quantization-aware training (QAT)** and **post-training quantization (PTQ)** both depend on exactly this operation. In ML compilers like Apache TVM and in ONNX Runtime, `clip + quantize` is a fused kernel that runs billions of times per inference batch. The TFLite converter applies it when producing `.tflite` models for mobile. Apple's Core ML uses it for model compression. Google's Gemma 2B uses INT8 quantization for a 4× size reduction with <1% accuracy loss.

In signal processing, clip-quantize is the mathematical model of an **analog-to-digital converter (ADC)**. A physical ADC samples a continuous voltage, clips it to the converter's dynamic range, and quantizes it into 2^N discrete levels (where N is the bit depth). A 16-bit audio ADC clips to ±V_ref and quantizes into 65,536 bins.

## How It Works

### Two-Step Operation

Given an input value `x`, a range `[lo, hi]`, and `B` bins:

**Step 1 — Clip** (clamp to range):

```
x_clamped = max(lo, min(hi, x))
```

Values outside `[lo, hi]` are saturated to the nearest bound. This is the **hard clip** — values are not reflected or wrapped, they're simply capped. The clipping error for out-of-range values is:

```
ε_clip(x) = x - x_clamped    (for |x| > boundary)
```

**Step 2 — Normalize** to [0, 1):

```
x_norm = (x_clamped - lo) / (hi - lo)
```

**Step 3 — Quantize** to B bins:

```
bin = floor(x_norm × B)
bin = min(bin, B - 1)    // handle x == hi edge case
```

**Step 4 — Dequantize** (reconstruct approximated value):

```
x_quantized = lo + (bin + 0.5) × step_size
```

Where `step_size = (hi - lo) / B`.

### Quantization Error Analysis

The quantization error for a value within range is:

```
ε_quant = |x - x_quantized| ≤ step_size / 2
```

This is **uniform quantization noise** — the error is uniformly distributed in [−Δ/2, +Δ/2] where Δ = step_size. The signal-to-quantization-noise ratio (SQNR) for a uniform quantizer with B bins is:

```
SQNR = 6.02 × log₂(B) + 1.76 dB
```

For each additional bit (doubling B), SQNR improves by ~6 dB. This is the foundational result of quantization theory (Bennett, 1948).

| Bins (B) | Bit Depth | Step Size (range=2) | Max Error | SQNR |
|-----------|-----------|--------------------|-----------|------|
| 4 | 2-bit | 0.500 | 0.250 | 13.8 dB |
| 16 | 4-bit | 0.125 | 0.063 | 25.9 dB |
| 256 | 8-bit | 0.0078 | 0.0039 | 50.0 dB |
| 65,536 | 16-bit | 0.000031 | 0.000015 | 98.1 dB |

### Slice Quantization

The `quantize_slice` function applies clip-quantize to an entire `&mut [f64]` in place. Each element is quantized and replaced with its dequantized value — simulating what happens when a float array passes through an ADC.

This is the exact operation performed by:

- `torch.quantization.quantize_per_tensor()` in PyTorch
- `tf.quantization.quantize()` in TensorFlow
- `numpy.digitize()` (without the clip step)

**Complexity:**

| Operation | Time | Space |
|-----------|------|-------|
| clip_quantize (single value) | O(1) | O(1) |
| quantize_slice (N elements) | O(N) | O(N) for bins vector |

## Quick Start

```rust
// The crate provides a demo binary with inline functions.
// The quantization logic:

fn clip_quantize(value: f64, lo: f64, hi: f64, bins: usize) -> usize {
    let clamped = value.clamp(lo, hi);
    let normalized = (clamped - lo) / (hi - lo);
    let bin = (normalized * bins as f64).floor() as usize;
    bin.min(bins - 1)
}

// Quantize a single value to range [-1, 2] with 4 bins
let bin = clip_quantize(0.7, -1.0, 2.0, 4);
// normalized: (0.7 - (-1)) / (2 - (-1)) = 1.7/3 ≈ 0.567
// bin: floor(0.567 × 4) = floor(2.267) = 2
assert_eq!(bin, 2);

// Out-of-range values are clipped
let clipped_high = clip_quantize(10.0, -1.0, 2.0, 4);
assert_eq!(clipped_high, 3);  // saturates to last bin
let clipped_low = clip_quantize(-5.0, -1.0, 2.0, 4);
assert_eq!(clipped_low, 0);   // saturates to first bin

// Run the full demo
// $ cargo run
// Input:  [-1.50, 0.20, 0.70, 1.10, 2.80, -0.30, 0.50, 1.90]
// Bins:   [0, 1, 2, 3, 3, 0, 2, 3]
// Output: [-0.62, 0.12, 0.88, 1.62, 1.62, -0.62, 0.88, 1.62]
```

## API

*Implemented as a demo binary with inline functions:*

| Function | Signature | Description |
|----------|-----------|-------------|
| `clip_quantize` | `(f64, Range<f64>, usize) → usize` | Clip then quantize a single value |
| `quantize_slice` | `(&mut [f64], Range<f64>, usize) → Vec<usize>` | In-place quantize + return bin indices |

## Architecture Notes

Clip-quantize is a primitive used across the SuperInstance ML toolchain for model compression and signal processing. In ML pipelines, it runs after training to convert float32 weights to int8 for deployment.

Within γ + η = C, quantization instantiates the conservation law as **information conservation with bounded loss**: the input signal (γ) is transformed to a discrete representation (η), and the total information (C) is approximately conserved — the error is bounded by Δ/2. The conservation invariant is the quantization error bound: information is not created (no false detail), only destroyed (clipping + rounding). The art is minimizing the destruction while maximizing the compression.

See the [architecture overview](https://github.com/casey-digennaro/clip-quantize/blob/main/ARCHITECTURE.md).

## References

1. Bennett, W.R. (1948). "Spectra of Quantized Signals." *Bell System Technical Journal*, 27(3), 446–472. (Foundational SQNR analysis)
2. Jacob, B. et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR 2018*. (QAT in practice)
3. Krishnamoorthi, R. (2018). "Quantizing Deep Convolutional Networks for Efficient Inference." *arXiv:1806.08342*. (PTQ survey)
4. Gray, R.M. & Neuhoff, D.L. (1998). "Quantization." *IEEE Transactions on Information Theory*, 44(6), 2325–2383. (Comprehensive survey)

## License

MIT
