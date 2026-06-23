use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::tensor::{numel, Tensor};

fn flatten_py_list(list: &Bound<'_, PyList>, out: &mut Vec<f32>) -> PyResult<()> {
    for item in list.iter() {
        if let Ok(nested) = item.downcast::<PyList>() {
            flatten_py_list(&nested, out)?;
        } else if let Ok(v) = item.extract::<f32>() {
            out.push(v);
        } else if let Ok(v) = item.extract::<i64>() {
            out.push(v as f32);
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "tensor() expects numeric scalars or nested lists",
            ));
        }
    }
    Ok(())
}

fn infer_shape(list: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
    let mut shape = vec![list.len()];
    let first = list.get_item(0)?;
    if let Ok(nested) = first.downcast::<PyList>() {
        let mut inner = infer_shape(&nested)?;
        inner.insert(0, shape[0]);
        return Ok(inner);
    }
    Ok(shape)
}

pub fn tensor_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if let Ok(v) = obj.extract::<f32>() {
        return Ok(Tensor::new(vec![v], vec![]));
    }
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Tensor::new(vec![v as f32], vec![]));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let shape = infer_shape(list)?;
        let mut data = Vec::new();
        flatten_py_list(list, &mut data)?;
        return Ok(Tensor::new(data, shape));
    }
    if let Ok(tuple) = obj.downcast::<pyo3::types::PyTuple>() {
        if tuple.len() == 2 {
            if let (Ok(data), Ok(batch_size)) = (
                tuple.get_item(0)?.downcast::<PyList>(),
                tuple.get_item(1)?.extract::<usize>(),
            ) {
                let inner_shape = infer_shape(data)?;
                let mut flat = Vec::new();
                flatten_py_list(data, &mut flat)?;
                let mut shape = vec![batch_size];
                shape.extend(inner_shape);
                if flat.len() != numel(&shape) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "tensor data length {} does not match shape {:?}",
                        flat.len(),
                        shape
                    )));
                }
                return Ok(Tensor::new(flat, shape));
            }
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "tensor() expects scalar, list, or (data, batch_size) tuple",
    ))
}

#[pyfunction]
#[pyo3(signature = (start, stop=None, step=1.0, dtype=None))]
pub fn arange(
    start: f32,
    stop: Option<f32>,
    step: f32,
    dtype: Option<&str>,
) -> PyResult<Tensor> {
    let _ = dtype;
    let (begin, end) = match stop {
        Some(s) => (start, s),
        None => (0.0, start),
    };
    if step == 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "arange step cannot be zero",
        ));
    }
    let mut data = Vec::new();
    let mut v = begin;
    if step > 0.0 {
        while v < end {
            data.push(v);
            v += step;
        }
    } else {
        while v > end {
            data.push(v);
            v += step;
        }
    }
    let len = data.len();
    Ok(Tensor::new(data, vec![len]))
}

#[pyfunction]
pub fn zeros(rows: usize, cols: usize) -> Tensor {
    Tensor::zeros(vec![rows, cols])
}

#[pyfunction]
#[pyo3(signature = (rows, cols, seed=None))]
pub fn randn(rows: usize, cols: usize, seed: Option<u64>) -> Tensor {
    let mut rng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = rows * cols;
    let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng) as f32).collect();
    Tensor::new(data, vec![rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_positive_step() {
        let t = arange(0.0, Some(5.0), 2.0, None).unwrap();
        assert_eq!(t.shape_vec(), vec![3]);
        assert_eq!(t.data(), &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn arange_single_arg() {
        let t = arange(4.0, None, 1.0, None).unwrap();
        assert_eq!(t.data(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn zeros_shape() {
        let t = zeros(2, 3);
        assert_eq!(t.shape_vec(), vec![2, 3]);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn randn_seeded_deterministic() {
        let a = randn(2, 2, Some(42));
        let b = randn(2, 2, Some(42));
        assert_eq!(a.data(), b.data());
    }
}
