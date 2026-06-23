use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;

use crate::autograd::{GradFn, track_unary};
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
}
