use pyo3::prelude::*;

use crate::autograd::{GradFn, track_unary};
use crate::broadcast::{broadcast_binary, tensor_from_any};
use crate::tensor::{Tensor, TensorCore};

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
    use crate::autograd::GradFn;
    use crate::broadcast::broadcast_binary;
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
