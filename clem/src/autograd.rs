use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::linalg::matmul as matmul_fn;
use crate::tensor::{broadcast_shapes, numel, Tensor, TensorCore, TensorId};

#[derive(Clone)]
pub enum GradFn {
    Leaf,
    Add { lhs: Tensor, rhs: Tensor },
    Sub { lhs: Tensor, rhs: Tensor },
    Mul { lhs: Tensor, rhs: Tensor },
    Div { lhs: Tensor, rhs: Tensor },
    Neg { input: Tensor },
    Exp { input: Tensor },
    Log { input: Tensor },
    Sin { input: Tensor },
    Cos { input: Tensor },
    Matmul { a: Tensor, b: Tensor },
    Softmax { input: Tensor, dim: usize },
    Gelu { input: Tensor },
}

pub fn accumulate_grad(cell: &std::cell::RefCell<Option<Vec<f32>>>, grad: &[f32]) {
    let mut g = cell.borrow_mut();
    match g.as_mut() {
        Some(existing) => {
            for (e, &v) in existing.iter_mut().zip(grad.iter()) {
                *e += v;
            }
        }
        None => {
            *g = Some(grad.to_vec());
        }
    }
}

pub fn track_unary(out: &Tensor, input: &Tensor, grad_fn: GradFn) {
    if *input.inner.requires_grad.borrow() || *out.inner.requires_grad.borrow() {
        *out.inner.grad_fn.borrow_mut() = Some(grad_fn);
    }
}

pub fn track_binary(out: &Tensor, lhs: &Tensor, rhs: &Tensor, make_fn: fn(Tensor, Tensor) -> GradFn) {
    if *lhs.inner.requires_grad.borrow()
        || *rhs.inner.requires_grad.borrow()
        || *out.inner.requires_grad.borrow()
    {
        *out.inner.grad_fn.borrow_mut() = Some(make_fn(lhs.clone(), rhs.clone()));
    }
}

pub fn track_matmul(out: &Tensor, a: &Tensor, b: &Tensor) {
    if *a.inner.requires_grad.borrow() || *b.inner.requires_grad.borrow() {
        *out.inner.grad_fn.borrow_mut() = Some(GradFn::Matmul {
            a: a.clone(),
            b: b.clone(),
        });
    }
}

fn sum_to_shape(grad: &[f32], grad_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    if grad_shape == target_shape {
        return grad.to_vec();
    }
    let out_n = numel(target_shape);
    let mut out = vec![0.0; out_n];
    let out_strides = crate::tensor::compute_strides(target_shape);
    let grad_strides = crate::tensor::compute_strides(grad_shape);

    for flat in 0..grad.len() {
        let mut rem = flat;
        let mut coords = vec![0usize; grad_shape.len()];
        for d in (0..grad_shape.len()).rev() {
            coords[d] = rem / grad_strides[d];
            rem %= grad_strides[d];
        }

        let offset = grad_shape.len().saturating_sub(target_shape.len());
        let mut target_coords = vec![0usize; target_shape.len()];
        for (ti, &dim) in target_shape.iter().enumerate() {
            let gi = ti + offset;
            if gi < grad_shape.len() {
                target_coords[ti] = if grad_shape[gi] == target_shape[ti] {
                    coords[gi]
                } else {
                    0
                };
            }
        }
        let dst: usize = target_coords
            .iter()
            .zip(out_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();
        out[dst] += grad[flat];
    }
    out
}

fn apply_grad_fn(node: &Tensor, grad: &[f32]) -> PyResult<Vec<(Tensor, Vec<f32>)>> {
    let gf = node.inner.grad_fn.borrow().clone();
    match gf {
        None | Some(GradFn::Leaf) => Ok(vec![]),
        Some(GradFn::Add { lhs, rhs }) => {
            let gl = sum_to_shape(grad, &node.shape_vec(), &lhs.shape_vec());
            let gr = sum_to_shape(grad, &node.shape_vec(), &rhs.shape_vec());
            Ok(vec![(lhs, gl), (rhs, gr)])
        }
        Some(GradFn::Sub { lhs, rhs }) => {
            let gl = sum_to_shape(grad, &node.shape_vec(), &lhs.shape_vec());
            let gr: Vec<f32> = sum_to_shape(grad, &node.shape_vec(), &rhs.shape_vec())
                .iter()
                .map(|x| -x)
                .collect();
            Ok(vec![(lhs, gl), (rhs, gr)])
        }
        Some(GradFn::Mul { lhs, rhs }) => {
            let gl = broadcast_mul_grad(grad, &node.shape_vec(), rhs.data(), &lhs.shape_vec(), &rhs.shape_vec());
            let gr = broadcast_mul_grad(grad, &node.shape_vec(), lhs.data(), &rhs.shape_vec(), &lhs.shape_vec());
            Ok(vec![(lhs, gl), (rhs, gr)])
        }
        Some(GradFn::Div { lhs, rhs }) => {
            let inv_rhs: Vec<f32> = rhs.data().iter().map(|&r| 1.0 / r).collect();
            let gl = broadcast_mul_grad(grad, &node.shape_vec(), &inv_rhs, &lhs.shape_vec(), &rhs.shape_vec());
            let gr = broadcast_div_rhs_grad(grad, &node.shape_vec(), &lhs, &rhs);
            Ok(vec![(lhs, gl), (rhs, gr)])
        }
        Some(GradFn::Neg { input }) => {
            let g: Vec<f32> = grad.iter().map(|x| -x).collect();
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
        Some(GradFn::Exp { input }) => {
            let mut g = grad.to_vec();
            for (gi, &ev) in g.iter_mut().zip(node.data().iter()) {
                *gi *= ev;
            }
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
        Some(GradFn::Log { input }) => {
            let mut g = grad.to_vec();
            for (gi, &iv) in g.iter_mut().zip(input.data().iter()) {
                *gi /= iv;
            }
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
        Some(GradFn::Sin { input }) => {
            let mut g = grad.to_vec();
            for (gi, &iv) in g.iter_mut().zip(input.data().iter()) {
                *gi *= iv.cos();
            }
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
        Some(GradFn::Cos { input }) => {
            let mut g = grad.to_vec();
            for (gi, &iv) in g.iter_mut().zip(input.data().iter()) {
                *gi *= -iv.sin();
            }
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
        Some(GradFn::Matmul { a, b }) => {
            let a_shape = a.shape_vec();
            let b_shape = b.shape_vec();
            if a_shape.len() == 2 && b_shape.len() == 2 {
                let grad_t = Tensor::from_core(TensorCore::new(grad.to_vec(), node.shape_vec()));
                let b_t = transpose_2d(&b)?;
                let a_grad = matmul_fn(&grad_t, &b_t)?;
                let a_t = transpose_2d(&a)?;
                let b_grad = matmul_fn(&a_t, &grad_t)?;
                return Ok(vec![
                    (a, a_grad.data().to_vec()),
                    (b, b_grad.data().to_vec()),
                ]);
            }
            Ok(vec![])
        }
        Some(GradFn::Softmax { input, dim }) => {
            let y = node.data();
            let mut g = grad.to_vec();
            let shape = node.shape_vec();
            let inner = shape[dim];
            let outer: usize = shape[..dim].iter().product();
            let block: usize = shape[dim + 1..].iter().product();
            for o in 0..outer {
                for b in 0..block {
                    let mut dot = 0.0f32;
                    for i in 0..inner {
                        let idx = (o * inner + i) * block + b;
                        dot += g[idx] * y[idx];
                    }
                    for i in 0..inner {
                        let idx = (o * inner + i) * block + b;
                        g[idx] = y[idx] * (g[idx] - dot);
                    }
                }
            }
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
        Some(GradFn::Gelu { input }) => {
            let g = grad
                .iter()
                .zip(input.data().iter())
                .map(|(&g, &x)| g * gelu_derivative(x))
                .collect::<Vec<_>>();
            let shape = input.shape_vec();
            Ok(vec![(input, sum_to_shape(&g, &node.shape_vec(), &shape))])
        }
    }
}

fn gelu_derivative(x: f32) -> f32 {
    let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
    let x3 = x.powi(3);
    let inner = sqrt_2_pi * (x + 0.044715 * x3);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner.powi(2);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2))
}

fn transpose_2d(t: &Tensor) -> PyResult<Tensor> {
    let shape = t.shape_vec();
    if shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "expected 2D tensor",
        ));
    }
    let (m, n) = (shape[0], shape[1]);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = t.data()[i * n + j];
        }
    }
    Ok(Tensor::from_core(TensorCore::new(out, vec![n, m])))
}

fn broadcast_mul_grad(
    grad: &[f32],
    out_shape: &[usize],
    other_data: &[f32],
    target_shape: &[usize],
    other_shape: &[usize],
) -> Vec<f32> {
    let broadcast = broadcast_shapes(out_shape, other_shape).unwrap_or_else(|_| out_shape.to_vec());
    let n = numel(&broadcast);
    let mut acc = vec![0.0; numel(target_shape)];

    let out_strides = crate::tensor::compute_strides(&broadcast);
    let other_strides = crate::tensor::compute_strides(other_shape);
    let target_strides = crate::tensor::compute_strides(target_shape);

    for flat in 0..n {
        let mut rem = flat;
        let mut out_coords = vec![0usize; broadcast.len()];
        for d in (0..broadcast.len()).rev() {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }

        let other_idx = broadcast_index(out_coords.as_slice(), other_shape, other_strides.as_slice());
        let target_idx = broadcast_index(out_coords.as_slice(), target_shape, target_strides.as_slice());
        acc[target_idx] += grad[flat] * other_data[other_idx.min(other_data.len() - 1)];
    }
    acc
}

fn broadcast_div_rhs_grad(grad: &[f32], out_shape: &[usize], lhs: &Tensor, rhs: &Tensor) -> Vec<f32> {
    let n = grad.len();
    let mut combined = vec![0.0; n];
    for i in 0..n {
        let l = lhs.data()[i % lhs.data().len()];
        let r = rhs.data()[i % rhs.data().len()];
        combined[i] = -grad[i] * l / (r * r);
    }
    sum_to_shape(&combined, out_shape, &rhs.shape_vec())
}

fn broadcast_index(out_coords: &[usize], shape: &[usize], strides: &[usize]) -> usize {
    let offset = out_coords.len().saturating_sub(shape.len());
    let mut idx = 0usize;
    for (si, &dim) in shape.iter().enumerate() {
        let oc = out_coords[si + offset];
        let c = if dim == 1 { 0 } else { oc };
        idx += c * strides[si];
    }
    idx
}

fn collect_nodes(root: &Tensor) -> Vec<Tensor> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![root.clone()];

    while let Some(node) = stack.pop() {
        if !visited.insert(node.inner.id) {
            continue;
        }
        if let Some(ref gf) = *node.inner.grad_fn.borrow() {
            let parents = match gf {
                GradFn::Leaf => vec![],
                GradFn::Add { lhs, rhs }
                | GradFn::Sub { lhs, rhs }
                | GradFn::Mul { lhs, rhs }
                | GradFn::Div { lhs, rhs } => vec![lhs.clone(), rhs.clone()],
                GradFn::Matmul { a, b } => vec![a.clone(), b.clone()],
                GradFn::Neg { input }
                | GradFn::Exp { input }
                | GradFn::Log { input }
                | GradFn::Sin { input }
                | GradFn::Cos { input }
                | GradFn::Softmax { input, .. }
                | GradFn::Gelu { input } => vec![input.clone()],
            };
            for p in parents {
                if !visited.contains(&p.inner.id) {
                    stack.push(p);
                }
            }
        }
        order.push(node);
    }
    order.reverse();
    order
}

pub fn backward(root: &Tensor) -> PyResult<()> {
    let nodes = collect_nodes(root);
    let mut grads: HashMap<TensorId, Vec<f32>> = HashMap::new();
    grads.insert(root.inner.id, vec![1.0; root.data().len().max(1)]);

    for node in nodes.iter().rev() {
        let grad = match grads.get(&node.inner.id) {
            Some(g) => g.clone(),
            None => continue,
        };
        if node.inner.grad_fn.borrow().is_none() {
            continue;
        }
        let parent_grads = apply_grad_fn(node, &grad)?;
        for (parent, g) in parent_grads {
            if *parent.inner.requires_grad.borrow() {
                match grads.get_mut(&parent.inner.id) {
                    Some(existing) => {
                        for (e, v) in existing.iter_mut().zip(g.iter()) {
                            *e += v;
                        }
                    }
                    None => {
                        grads.insert(parent.inner.id, g.clone());
                    }
                }
                parent.inner.accumulate_grad(&grads[&parent.inner.id]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::exp;
    use crate::tensor::TensorCore;

    #[test]
    fn exp_backward() {
        let x = Tensor::from_core(TensorCore::new(vec![1.0], vec![1]));
        x.inner.set_requires_grad(true);
        let y = exp(&x).unwrap();
        backward(&y).unwrap();
        let g = x.inner.grad.borrow();
        assert!(g.is_some());
        let grad = g.as_ref().unwrap();
        assert!((grad[0] - std::f32::consts::E).abs() < 1e-3);
    }
}
