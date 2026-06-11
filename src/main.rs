/// clip-quantize: Clips values to a range and quantizes them into discrete bins.
use std::ops::Range;

fn clip_quantize(value: f64, range: Range<f64>, bins: usize) -> usize {
    let clamped = value.clamp(range.start, range.end);
    let normalized = (clamped - range.start) / (range.end - range.start);
    let bin = (normalized * bins as f64).floor() as usize;
    bin.min(bins - 1)
}

fn quantize_slice(data: &mut [f64], range: Range<f64>, bins: usize) -> Vec<usize> {
    data.iter_mut()
        .map(|v| {
            let bin = clip_quantize(*v, range.clone(), bins);
            let step = (range.end - range.start) / bins as f64;
            *v = range.start + (bin as f64 + 0.5) * step;
            bin
        })
        .collect()
}

fn main() {
    let mut data = vec![-1.5, 0.2, 0.7, 1.1, 2.8, -0.3, 0.5, 1.9];
    println!("Input:  {data:.2?}");

    let bins = quantize_slice(&mut data, -1.0..2.0, 4);
    println!("Bins:   {bins:?}");
    println!("Output: {data:.2?}");
}
