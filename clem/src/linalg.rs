use pyo3::prelude::*;

use crate::autograd::track_matmul;
use crate::tensor::{numel, Tensor, TensorCore};

pub fn matmul(a: &Tensor, b: &Tensor) -> PyResult<Tensor> {
    let a_shape = a.shape_vec();
    let b_shape = b.shape_vec();

    let out = if a_shape.len() == 2 && b_shape.len() == 2 {
        matmul_2d(a, b)?
    } else if a_shape.len() == 3 && b_shape.len() == 3 {
        matmul_batched(a, b)?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "matmul supports 2D or batched 3D tensors",
        ));
    };

    track_matmul(&out, a, b);
    Ok(out)
}

fn matmul_2d(a: &Tensor, b: &Tensor) -> PyResult<Tensor> {
    let (m, k) = (a.shape_vec()[0], a.shape_vec()[1]);
    let (k2, n) = (b.shape_vec()[0], b.shape_vec()[1]);
    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "matmul shape mismatch: ({}, {}) @ ({}, {})",
            m, k, k2, n
        )));
    }
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a.data()[i * k + p] * b.data()[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Ok(Tensor::from_core(TensorCore::new(out, vec![m, n])))
}

fn matmul_batched(a: &Tensor, b: &Tensor) -> PyResult<Tensor> {
    let a_shape = a.shape_vec();
    let b_shape = b.shape_vec();
    let batch = a_shape[0];
    if batch != b_shape[0] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "batch size mismatch",
        ));
    }
    let m = a_shape[1];
    let k = a_shape[2];
    let k2 = b_shape[1];
    let n = b_shape[2];
    if k != k2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "inner dimension mismatch for batched matmul",
        ));
    }
    let mut out = vec![0.0; batch * m * n];
    let a_stride = m * k;
    let b_stride = k * n;
    for batch_idx in 0..batch {
        let a_off = batch_idx * a_stride;
        let b_off = batch_idx * b_stride;
        let o_off = batch_idx * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a.data()[a_off + i * k + p] * b.data()[b_off + p * n + j];
                }
                out[o_off + i * n + j] = sum;
            }
        }
    }
    Ok(Tensor::from_core(TensorCore::new(out, vec![batch, m, n])))
}

pub fn transpose(a: &Tensor, dim0: usize, dim1: usize) -> PyResult<Tensor> {
    let shape = a.shape_vec();
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "transpose dimensions out of range",
        ));
    }
    let mut new_shape = shape.clone();
    new_shape.swap(dim0, dim1);
    let n = numel(&shape);
    let mut out = vec![0.0; n];
    let out_strides = crate::tensor::compute_strides(&new_shape);

    for flat in 0..n {
        let coords = crate::tensor::unravel_index(flat, &shape);
        let mut swapped = coords.clone();
        swapped.swap(dim0, dim1);
        let dst_idx: usize = swapped
            .iter()
            .zip(out_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();
        out[dst_idx] = a.data()[flat];
    }

    Ok(Tensor::from_core(TensorCore::new(out, new_shape)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorCore;

    #[test]
    fn matmul_2x2() {
        let a = Tensor::from_core(TensorCore::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        let b = Tensor::from_core(TensorCore::new(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]));
        let out = matmul_2d(&a, &b).unwrap();
        assert_eq!(out.data(), &[4.0, 4.0, 10.0, 8.0]);
    }

    #[test]
    fn batched_matmul() {
        let a = Tensor::from_core(TensorCore::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]));
        let b = Tensor::from_core(TensorCore::new(vec![1.0, 0.0, 0.0, 1.0], vec![1, 2, 2]));
        let out = matmul_batched(&a, &b).unwrap();
        assert_eq!(out.shape_vec(), vec![1, 2, 2]);
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
