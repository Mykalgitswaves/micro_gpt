use pyo3::prelude::*;

use crate::autograd::{GradFn, track_binary, track_unary};
use crate::tensor::{broadcast_shapes, compute_strides, numel, Tensor, TensorCore};

fn tensor_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if let Ok(t) = obj.extract::<Tensor>() {
        Ok(t)
    } else if let Ok(v) = obj.extract::<f32>() {
        Ok(Tensor::new(vec![v], vec![]))
    } else if let Ok(v) = obj.extract::<i64>() {
        Ok(Tensor::new(vec![v as f32], vec![]))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected Tensor or numeric scalar",
        ))
    }
}

fn coords_to_index(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum()
}

fn unravel(flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut coords = vec![0; shape.len()];
    let mut rem = flat;
    for d in 0..shape.len() {
        coords[d] = rem / strides[d];
        rem %= strides[d];
    }
    coords
}

fn broadcast_input_index(out_coords: &[usize], out_shape: &[usize], in_shape: &[usize]) -> usize {
    let offset = out_shape.len().saturating_sub(in_shape.len());
    let mut in_coords = vec![0; in_shape.len()];
    for (i, &dim) in in_shape.iter().enumerate() {
        let oc = out_coords[i + offset];
        in_coords[i] = if dim == 1 { 0 } else { oc };
    }
    coords_to_index(&in_coords, &compute_strides(in_shape))
}

fn broadcast_binary<F>(a: &Tensor, b: &Tensor, op: F, grad_fn: fn(Tensor, Tensor) -> GradFn) -> PyResult<Tensor>
where
    F: Fn(f32, f32) -> f32,
{
    let a_shape = a.shape_vec();
    let b_shape = b.shape_vec();
    let out_shape = broadcast_shapes(&a_shape, &b_shape)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    let out_n = numel(&out_shape);
    let mut out_data = vec![0.0; out_n];

    for flat in 0..out_n {
        let out_coords = unravel(flat, &out_shape);
        let a_idx = if a_shape.is_empty() {
            0
        } else {
            broadcast_input_index(&out_coords, &out_shape, &a_shape)
        };
        let b_idx = if b_shape.is_empty() {
            0
        } else {
            broadcast_input_index(&out_coords, &out_shape, &b_shape)
        };
        out_data[flat] = op(a.data()[a_idx], b.data()[b_idx]);
    }

    let out = Tensor::from_core(TensorCore::new(out_data, out_shape));
    track_binary(&out, a, b, grad_fn);
    Ok(out)
}

fn map_unary(a: &Tensor, f: fn(f32) -> f32, grad: GradFn) -> PyResult<Tensor> {
    let data: Vec<f32> = a.data().iter().map(|&x| f(x)).collect();
    let out = Tensor::from_core(TensorCore::new(data, a.shape_vec()));
    track_unary(&out, a, grad);
    Ok(out)
}

pub fn exp(a: &Tensor) -> PyResult<Tensor> {
    let a_clone = a.clone();
    map_unary(a, |x| x.exp(), GradFn::Exp { input: a_clone })
}

pub fn log(a: &Tensor) -> PyResult<Tensor> {
    let a_clone = a.clone();
    map_unary(a, |x| x.ln(), GradFn::Log { input: a_clone })
}

pub fn sin(a: &Tensor) -> PyResult<Tensor> {
    let a_clone = a.clone();
    map_unary(a, |x| x.sin(), GradFn::Sin { input: a_clone })
}

pub fn cos(a: &Tensor) -> PyResult<Tensor> {
    let a_clone = a.clone();
    map_unary(a, |x| x.cos(), GradFn::Cos { input: a_clone })
}

pub fn neg(a: &Tensor) -> PyResult<Tensor> {
    let a_clone = a.clone();
    map_unary(a, |x| -x, GradFn::Neg { input: a_clone })
}

pub fn add(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let b = tensor_from_any(other)?;
    broadcast_binary(a, &b, |x, y| x + y, |lhs, rhs| GradFn::Add { lhs, rhs })
}

pub fn sub(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let b = tensor_from_any(other)?;
    broadcast_binary(a, &b, |x, y| x - y, |lhs, rhs| GradFn::Sub { lhs, rhs })
}

pub fn mul(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let b = tensor_from_any(other)?;
    broadcast_binary(a, &b, |x, y| x * y, |lhs, rhs| GradFn::Mul { lhs, rhs })
}

pub fn div(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let b = tensor_from_any(other)?;
    broadcast_binary(a, &b, |x, y| x / y, |lhs, rhs| GradFn::Div { lhs, rhs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorCore;

    #[test]
    fn unary_exp() {
        let t = Tensor::from_core(TensorCore::new(vec![0.0, 1.0], vec![2]));
        let out = exp(&t).unwrap();
        assert!((out.data()[0] - 1.0).abs() < 1e-5);
        assert!((out.data()[1] - std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn broadcast_mul_pos_div_term() {
        let pos = Tensor::from_core(TensorCore::new(
            (0..256).map(|i| i as f32).collect(),
            vec![256, 1],
        ));
        let div_term = Tensor::from_core(TensorCore::new(
            (0..32).map(|i| i as f32).collect(),
            vec![32],
        ));
        let out = broadcast_binary(&pos, &div_term, |a, b| a * b, |l, r| GradFn::Mul { lhs: l, rhs: r })
            .unwrap();
        assert_eq!(out.shape_vec(), vec![256, 32]);
    }

    #[test]
    fn add_broadcast() {
        let a = Tensor::from_core(TensorCore::new(vec![1.0, 2.0, 3.0], vec![3, 1]));
        let b = Tensor::from_core(TensorCore::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]));
        let out = broadcast_binary(&a, &b, |x, y| x + y, |l, r| GradFn::Add { lhs: l, rhs: r }).unwrap();
        assert_eq!(out.shape_vec(), vec![3, 4]);
        assert_eq!(out.data()[0], 11.0);
        assert_eq!(out.data()[4], 12.0);
    }
}
