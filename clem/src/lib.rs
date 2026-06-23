mod autograd;
mod creation;
mod indexing;
mod linalg;
mod nn;
mod ops;
mod tensor;

use pyo3::prelude::*;

use creation::{arange, randn, tensor_from_py, zeros};
use nn::{dropout, gelu, layer_norm};
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
    m.add_function(wrap_pyfunction!(dropout, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_fn, m)?)?;
    m.add_function(wrap_pyfunction!(gelu_fn, m)?)?;
    Ok(())
}
