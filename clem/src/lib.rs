mod autograd;
mod broadcast;
mod compare;
mod creation;
mod indexing;
mod linalg;
mod mask;
mod nn;
mod ops;
mod tensor;

use pyo3::prelude::*;

use compare::{eq as eq_op, ge as ge_op, gt as gt_op, le as le_op, lt as lt_op, ne as ne_op};
use creation::{arange, randn, tensor_from_py, zeros};
use mask::{masked_fill, where_fn};
use nn::{cross_entropy, dropout, gelu, layer_norm};
use ops::{cos as cos_op, exp as exp_op, log as log_op, sin as sin_op};
use tensor::Tensor;

#[pyfunction(name = "tensor")]
fn make_tensor(data: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    tensor_from_py(data)
}

#[pyfunction]
fn layer_norm_fn(tensor: &Tensor, normalized_shape: Vec<usize>) -> PyResult<Tensor> {
    layer_norm(tensor, normalized_shape)
}

#[pyfunction]
fn gelu_fn(tensor: &Tensor) -> PyResult<Tensor> {
    gelu(tensor)
}

#[pyfunction]
fn exp(tensor: &Tensor) -> PyResult<Tensor> {
    exp_op(tensor)
}

#[pyfunction]
fn log(tensor: &Tensor) -> PyResult<Tensor> {
    log_op(tensor)
}

#[pyfunction]
fn sin(tensor: &Tensor) -> PyResult<Tensor> {
    sin_op(tensor)
}

#[pyfunction]
fn cos(tensor: &Tensor) -> PyResult<Tensor> {
    cos_op(tensor)
}

#[pyfunction(name = "eq")]
fn eq_py(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    eq_op(a, other)
}

#[pyfunction(name = "ne")]
fn ne_py(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    ne_op(a, other)
}

#[pyfunction(name = "gt")]
fn gt_py(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    gt_op(a, other)
}

#[pyfunction(name = "ge")]
fn ge_py(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    ge_op(a, other)
}

#[pyfunction(name = "lt")]
fn lt_py(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    lt_op(a, other)
}

#[pyfunction(name = "le")]
fn le_py(a: &Tensor, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    le_op(a, other)
}

#[pyfunction(name = "where")]
fn where_py(
    condition: &Tensor,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<Tensor> {
    where_fn(condition, x, y)
}

#[pyfunction]
fn masked_fill_fn(tensor: &Tensor, mask: &Tensor, value: f32) -> PyResult<Tensor> {
    masked_fill(tensor, mask, value)
}

#[pymodule]
fn clem(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    m.add_function(wrap_pyfunction!(make_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(eq_py, m)?)?;
    m.add_function(wrap_pyfunction!(ne_py, m)?)?;
    m.add_function(wrap_pyfunction!(gt_py, m)?)?;
    m.add_function(wrap_pyfunction!(ge_py, m)?)?;
    m.add_function(wrap_pyfunction!(lt_py, m)?)?;
    m.add_function(wrap_pyfunction!(le_py, m)?)?;
    m.add_function(wrap_pyfunction!(where_py, m)?)?;
    m.add_function(wrap_pyfunction!(masked_fill_fn, m)?)?;
    m.add_function(wrap_pyfunction!(dropout, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(gelu_fn, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    Ok(())
}
