use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;

use crate::autograd::{track_cross_entropy, GradFn, track_unary};
use crate::tensor::{numel, Tensor, TensorCore};

pub fn softmax(a: &Tensor, dim: isize) -> PyResult<Tensor> {
    let shape = a.shape_vec();
    if shape.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "softmax requires at least 1D tensor",
        ));
    }
    let ndim = shape.len();
    let mut d = dim;
    if d < 0 {
        d += ndim as isize;
    }
    if d < 0 || d >= ndim as isize {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "softmax dimension out of range",
        ));
    }
    let dim = d as usize;

    let mut out = a.data().to_vec();
    let inner = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let block: usize = shape[dim + 1..].iter().product();

    for o in 0..outer {
        for b in 0..block {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..inner {
                let idx = (o * inner + i) * block + b;
                max_val = max_val.max(out[idx]);
            }
            let mut sum = 0.0f32;
            for i in 0..inner {
                let idx = (o * inner + i) * block + b;
                out[idx] = (out[idx] - max_val).exp();
                sum += out[idx];
            }
            for i in 0..inner {
                let idx = (o * inner + i) * block + b;
                out[idx] /= sum;
            }
        }
    }

    let a_clone = a.clone();
    let out_t = Tensor::from_core(TensorCore::new(out, shape.clone()));
    track_unary(
        &out_t,
        a,
        GradFn::Softmax {
            input: a_clone,
            dim,
        },
    );
    Ok(out_t)
}

pub fn layer_norm(a: &Tensor, normalized_shape: Vec<usize>) -> PyResult<Tensor> {
    let shape = a.shape_vec();
    let norm_n: usize = normalized_shape.iter().product();
    if norm_n == 0 || shape.len() < normalized_shape.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "invalid normalized_shape for layer_norm",
        ));
    }
    let outer: usize = numel(&shape) / norm_n;
    let mut out = a.data().to_vec();
    for block in 0..outer {
        let start = block * norm_n;
        let end = start + norm_n;
        let slice = &a.data()[start..end];
        let mean = slice.iter().sum::<f32>() / norm_n as f32;
        let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / norm_n as f32;
        let std = (var + 1e-5).sqrt();
        for (i, x) in slice.iter().enumerate() {
            out[start + i] = (x - mean) / std;
        }
    }
    Ok(Tensor::from_core(TensorCore::new(out, shape)))
}

pub fn gelu(a: &Tensor) -> PyResult<Tensor> {
    let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
    let data: Vec<f32> = a
        .data()
        .iter()
        .map(|&x| {
            let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
            x * 0.5 * (1.0 + inner.tanh())
        })
        .collect();
    let a_clone = a.clone();
    let out = Tensor::from_core(TensorCore::new(data, a.shape_vec()));
    track_unary(&out, a, GradFn::Gelu { input: a_clone });
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (tensor, p=0.5, seed=None))]
pub fn dropout(tensor: &Tensor, p: f32, seed: Option<u64>) -> PyResult<Tensor> {
    if p < 0.0 || p >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "dropout probability must be in [0, 1)",
        ));
    }
    let scale = 1.0 / (1.0 - p);
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let data: Vec<f32> = tensor
        .data()
        .iter()
        .map(|&x| {
            if rng.gen::<f32>() < p {
                0.0
            } else {
                x * scale
            }
        })
        .collect();
    Ok(Tensor::from_core(TensorCore::new(data, tensor.shape_vec())))
}

/// Cross-entropy loss for next-token classification.
///
/// - `logits`: model output scores, shape `(N, C)` — from `output_head`, one row per token position.
/// - `targets`: ground-truth class indices, shape `(N,)` — token IDs (e.g. shifted `idx_tensor`).
/// - `ignore_index`: skip positions where target equals this value (e.g. padding token id).
///
/// Returns a scalar mean loss over non-ignored positions.
#[pyfunction]
#[pyo3(signature = (logits, targets, ignore_index=None))]
pub fn cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: Option<i32>,
) -> PyResult<Tensor> {
    let logit_shape = logits.shape_vec();
    if logit_shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cross_entropy expects 2D logits (N, C)",
        ));
    }
    let (n, c) = (logit_shape[0], logit_shape[1]);
    if numel(&targets.shape_vec()) != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "targets must have N={} elements, got shape {:?}",
            n,
            targets.shape_vec()
        )));
    }

    let logits_data = logits.data();
    let targets_data = targets.data();
    let mut loss_sum = 0.0f32;
    let mut num_valid = 0usize;

    for row in 0..n {
        let target = targets_data[row] as i32;
        if ignore_index.is_some_and(|ig| target == ig) {
            continue;
        }
        let target_class = target as usize;
        if target_class >= c {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "target index {} out of range for {} classes",
                target_class, c
            )));
        }

        let row_start = row * c;
        let row_slice = &logits_data[row_start..row_start + c];
        let max = row_slice
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = max + row_slice.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
        loss_sum -= row_slice[target_class] - log_sum_exp;
        num_valid += 1;
    }

    if num_valid == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cross_entropy: no valid targets after applying ignore_index",
        ));
    }

    let loss = loss_sum / num_valid as f32;
    let out = Tensor::from_core(TensorCore::new(vec![loss], vec![]));
    track_cross_entropy(&out, logits, targets, ignore_index, num_valid);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorCore;

    #[test]
    fn softmax_rows_sum_to_one() {
        let t = Tensor::from_core(TensorCore::new(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]));
        let out = softmax(&t, -1).unwrap();
        let row0: f32 = out.data()[0..3].iter().sum();
        let row1: f32 = out.data()[3..6].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-5);
        assert!((row1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_zero_mean() {
        let t = Tensor::from_core(TensorCore::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        let out = layer_norm(&t, vec![2]).unwrap();
        let mean = out.data().iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn gelu_at_zero() {
        let t = Tensor::from_core(TensorCore::new(vec![0.0], vec![1]));
        let out = gelu(&t).unwrap();
        assert!(out.data()[0].abs() < 1e-5);
    }

    #[test]
    fn dropout_seeded() {
        let t = Tensor::from_core(TensorCore::new(vec![1.0; 4], vec![4]));
        let a = dropout(&t, 0.5, Some(123)).unwrap();
        let b = dropout(&t, 0.5, Some(123)).unwrap();
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn cross_entropy_known_value() {
        // two classes, one sample: logits favor class 1, target is class 1 -> low loss
        let logits = Tensor::from_core(TensorCore::new(vec![0.0, 2.0], vec![1, 2]));
        let targets = Tensor::from_core(TensorCore::new(vec![1.0], vec![1]));
        let loss = cross_entropy(&logits, &targets, None).unwrap();
        assert!(loss.data()[0] < 0.5);
    }
}
