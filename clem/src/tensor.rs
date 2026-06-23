use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::autograd::{accumulate_grad, GradFn};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TensorId(pub u64);

fn new_id() -> TensorId {
    TensorId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0; max_len];
    for i in 0..max_len {
        let dim_a = if i < a.len() {
            a[a.len() - 1 - i]
        } else {
            1
        };
        let dim_b = if i < b.len() {
            b[b.len() - 1 - i]
        } else {
            1
        };
        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return Err(format!("cannot broadcast shapes {:?} and {:?}", a, b));
        }
        result[max_len - 1 - i] = dim_a.max(dim_b);
    }
    Ok(result)
}

#[derive(Clone)]
pub struct TensorCore {
    pub id: TensorId,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub grad: RefCell<Option<Vec<f32>>>,
    pub requires_grad: RefCell<bool>,
    pub grad_fn: RefCell<Option<GradFn>>,
}

impl TensorCore {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            id: new_id(),
            data,
            shape,
            strides,
            grad: RefCell::new(None),
            requires_grad: RefCell::new(false),
            grad_fn: RefCell::new(None),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = numel(&shape);
        Self::new(vec![0.0; n], shape)
    }

    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<TensorCore, String> {
        if numel(&new_shape) != self.numel() {
            return Err(format!("cannot reshape {:?} to {:?}", self.shape, new_shape));
        }
        Ok(TensorCore::new(self.data.clone(), new_shape))
    }

    pub fn set_requires_grad(&self, value: bool) {
        *self.requires_grad.borrow_mut() = value;
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }

    pub fn accumulate_grad(&self, grad: &[f32]) {
        accumulate_grad(&self.grad, grad);
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Tensor {
    pub inner: Rc<TensorCore>,
}

impl Tensor {
    pub fn from_core(core: TensorCore) -> Self {
        Self {
            inner: Rc::new(core),
        }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_core(TensorCore::new(data, shape))
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::from_core(TensorCore::zeros(shape))
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    pub fn data(&self) -> &[f32] {
        &self.inner.data
    }

    pub fn set_grad_fn(&self, grad_fn: GradFn) {
        *self.inner.grad_fn.borrow_mut() = Some(grad_fn);
    }
}

#[pymethods]
impl Tensor {
    #[getter]
    fn shape(&self) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let items: Vec<_> = self.inner.shape.iter().map(|&d| d).collect();
            Ok(PyTuple::new(py, items)?.into())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, data=[{}...])",
            self.inner.shape,
            self.inner
                .data
                .iter()
                .take(6)
                .map(|x| format!("{:.4}", x))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn __len__(&self) -> usize {
        if self.inner.shape.is_empty() {
            1
        } else {
            self.inner.shape[0]
        }
    }

    #[pyo3(signature = (*dims))]
    fn reshape(&self, dims: &Bound<'_, PyTuple>) -> PyResult<Tensor> {
        let new_shape: Vec<usize> = dims
            .iter()
            .map(|d| d.extract::<usize>())
            .collect::<PyResult<_>>()?;
        self.inner
            .reshape(new_shape)
            .map(Tensor::from_core)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn requires_grad_(&self, value: bool) {
        self.inner.set_requires_grad(value);
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        *self.inner.requires_grad.borrow()
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<Tensor>> {
        let g = self.inner.grad.borrow();
        match g.as_ref() {
            Some(data) => Ok(Some(Tensor::new(data.clone(), self.inner.shape.clone()))),
            None => Ok(None),
        }
    }

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn backward(&self) -> PyResult<()> {
        crate::autograd::backward(self)
    }

    #[pyo3(signature = (dim0=None, dim1=None))]
    fn transpose(&self, dim0: Option<usize>, dim1: Option<usize>) -> PyResult<Tensor> {
        crate::linalg::transpose(self, dim0.unwrap_or(0), dim1.unwrap_or(1))
    }

    fn matmul(&self, other: &Tensor) -> PyResult<Tensor> {
        crate::linalg::matmul(self, other)
    }

    fn softmax(&self, dim: isize) -> PyResult<Tensor> {
        crate::nn::softmax(self, dim)
    }

    fn __neg__(&self) -> PyResult<Tensor> {
        crate::ops::neg(self)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        crate::ops::add(self, other)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        crate::ops::add(self, other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        crate::ops::sub(self, other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        crate::ops::mul(self, other)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        crate::ops::mul(self, other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        crate::ops::div(self, other)
    }

    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        crate::linalg::matmul(self, other)
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        crate::indexing::getitem(self, key)
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::indexing::setitem(self, key, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_strides_row_major() {
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[4, 2, 3]), vec![6, 3, 1]);
    }

    #[test]
    fn broadcast_shapes_numpy_rules() {
        assert_eq!(broadcast_shapes(&[3, 1], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[256, 1], &[64]).unwrap(), vec![256, 64]);
        assert!(broadcast_shapes(&[3, 4], &[2, 4]).is_err());
    }

    #[test]
    fn reshape_validates_element_count() {
        let core = TensorCore::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(core.reshape(vec![4]).is_ok());
        assert!(core.reshape(vec![3]).is_err());
    }

    #[test]
    fn numel_product() {
        assert_eq!(numel(&[2, 3, 4]), 24);
        assert_eq!(numel(&[]), 1);
    }
}
