use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use crate::tensor::{numel, Tensor, TensorCore};

fn parse_slice_for_dim(slice: &Bound<'_, PySlice>, dim: usize) -> PyResult<Vec<usize>> {
    let len = dim as isize;
    let indices = slice.indices(len)?;
    let step = indices.step;
    if step == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "slice step cannot be zero",
        ));
    }
    let mut coords = Vec::new();
    let mut i = indices.start;
    if step > 0 {
        while i < indices.stop {
            if i >= 0 && i < len {
                coords.push(i as usize);
            }
            i += step;
        }
    } else {
        while i > indices.stop {
            if i >= 0 && i < len {
                coords.push(i as usize);
            }
            i += step;
        }
    }
    Ok(coords)
}

fn normalize_index(idx: isize, dim: usize) -> PyResult<usize> {
    let d = dim as isize;
    let mut i = idx;
    if i < 0 {
        i += d;
    }
    if i < 0 || i >= d {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
            "index {} out of range for dimension size {}",
            idx, dim
        )));
    }
    Ok(i as usize)
}

pub fn getitem(tensor: &Tensor, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        if let Ok(t) = key.extract::<Tensor>() {
            let out = advanced_index(tensor, &t)?;
            return Ok(out.into_py(py));
        }
        if let Ok(idx) = key.extract::<isize>() {
            let out = getitem_int(tensor, idx)?;
            return Ok(out.into_py(py));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let out = getitem_slice_1d(tensor, slice)?;
            return Ok(out.into_py(py));
        }
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            let out = getitem_tuple(tensor, tuple)?;
            return Ok(out.into_py(py));
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "invalid index type",
        ))
    })
}

fn getitem_int(tensor: &Tensor, idx: isize) -> PyResult<Tensor> {
    let shape = tensor.shape_vec();
    if shape.len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "integer indexing on non-1D tensor requires tuple indexing",
        ));
    }
    let i = normalize_index(idx, shape[0])?;
    Ok(Tensor::new(vec![tensor.data()[i]], vec![]))
}

fn getitem_slice_1d(tensor: &Tensor, slice: &Bound<'_, PySlice>) -> PyResult<Tensor> {
    let shape = tensor.shape_vec();
    if shape.len() != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "1D slice on multi-dim tensor requires tuple indexing",
        ));
    }
    let coords = parse_slice_for_dim(slice, shape[0])?;
    let data: Vec<f32> = coords.iter().map(|&i| tensor.data()[i]).collect();
    Ok(Tensor::new(data, vec![coords.len()]))
}

fn getitem_tuple(tensor: &Tensor, tuple: &Bound<'_, PyTuple>) -> PyResult<Tensor> {
    let shape = tensor.shape_vec();
    if tuple.len() > shape.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "too many indices",
        ));
    }

    let mut dim_coords: Vec<Vec<usize>> = shape.iter().map(|&d| (0..d).collect()).collect();

    for (dim, item) in tuple.iter().enumerate() {
        if item.is_none() {
            continue;
        }
        if let Ok(idx) = item.extract::<isize>() {
            dim_coords[dim] = vec![normalize_index(idx, shape[dim])?];
        } else if let Ok(slice) = item.downcast::<PySlice>() {
            dim_coords[dim] = parse_slice_for_dim(slice, shape[dim])?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "index must be int or slice",
            ));
        }
    }

    if shape.len() == 2 {
        let rows = &dim_coords[0];
        let cols = &dim_coords[1];
        let mut data = Vec::with_capacity(rows.len() * cols.len());
        for &r in rows {
            for &c in cols {
                data.push(tensor.data()[r * shape[1] + c]);
            }
        }
        return Ok(Tensor::new(data, vec![rows.len(), cols.len()]));
    }

    if shape.len() == 1 {
        return Ok(Tensor::new(
            dim_coords[0]
                .iter()
                .map(|&i| tensor.data()[i])
                .collect(),
            vec![dim_coords[0].len()],
        ));
    }

    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "tuple indexing only implemented for 1D and 2D tensors",
    ))
}

fn advanced_index(table: &Tensor, indices: &Tensor) -> PyResult<Tensor> {
    let table_shape = table.shape_vec();
    if table_shape.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "advanced indexing only supported for 2D embedding tables",
        ));
    }
    let idx_shape = indices.shape_vec();
    let vocab = table_shape[0];
    let d_model = table_shape[1];
    let out_n = numel(&idx_shape);
    let mut out_data = vec![0.0; out_n * d_model];

    for flat in 0..out_n {
        let row = indices.data()[flat] as usize;
        if row >= vocab {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} out of range for vocab size {}",
                row, vocab
            )));
        }
        let src_start = row * d_model;
        let dst_start = flat * d_model;
        out_data[dst_start..dst_start + d_model]
            .copy_from_slice(&table.data()[src_start..src_start + d_model]);
    }

    let mut out_shape = idx_shape;
    out_shape.push(d_model);
    Ok(Tensor::new(out_data, out_shape))
}

pub fn setitem(tensor: &mut Tensor, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let val = if let Ok(t) = value.extract::<Tensor>() {
        t
    } else {
        crate::creation::tensor_from_py(value)?
    };

    if let Ok(tuple) = key.downcast::<PyTuple>() {
        return setitem_tuple(tensor, tuple, &val);
    }
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "setitem only supports tuple indexing",
    ))
}

fn setitem_tuple(tensor: &mut Tensor, tuple: &Bound<'_, PyTuple>, value: &Tensor) -> PyResult<()> {
    let shape = tensor.shape_vec();
    if shape.len() != 2 || tuple.len() != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "setitem tuple indexing requires 2 indices for 2D tensor",
        ));
    }

    let row_item = tuple.get_item(0)?;
    let col_item = tuple.get_item(1)?;

    let row_coords = if row_item.is_none() {
        (0..shape[0]).collect()
    } else if let Ok(slice) = row_item.downcast::<PySlice>() {
        parse_slice_for_dim(slice, shape[0])?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "row index must be slice or None",
        ));
    };

    let col_coords = if let Ok(slice) = col_item.downcast::<PySlice>() {
        parse_slice_for_dim(slice, shape[1])?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "col index must be slice",
        ));
    };

    let expected_rows = row_coords.len();
    let expected_cols = col_coords.len();
    let val_shape = value.shape_vec();
    if val_shape != vec![expected_rows, expected_cols] {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "value shape {:?} does not match slice ({}, {})",
            val_shape, expected_rows, expected_cols
        )));
    }

    if let Some(core_mut) = std::rc::Rc::get_mut(&mut tensor.inner) {
        for (ri, &r) in row_coords.iter().enumerate() {
            for (ci, &c) in col_coords.iter().enumerate() {
                let dst = r * shape[1] + c;
                let src = ri * expected_cols + ci;
                core_mut.data[dst] = value.data()[src];
            }
        }
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "cannot mutate tensor with shared references",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PySlice;
    use crate::tensor::TensorCore;

    #[test]
    fn advanced_index_embedding() {
        let table = Tensor::from_core(TensorCore::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ));
        let idx = Tensor::from_core(TensorCore::new(vec![0.0, 2.0], vec![2]));
        let out = advanced_index(&table, &idx).unwrap();
        assert_eq!(out.shape_vec(), vec![2, 2]);
        assert_eq!(out.data(), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn setitem_even_columns() {
        let mut t = Tensor::from_core(TensorCore::new(vec![0.0; 12], vec![3, 4]));
        let val = Tensor::from_core(TensorCore::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ));
        let row_coords: Vec<usize> = (0..3).collect();
        let col_coords: Vec<usize> = (0..4).step_by(2).collect();
        if let Some(core_mut) = std::rc::Rc::get_mut(&mut t.inner) {
            for (ri, &r) in row_coords.iter().enumerate() {
                for (ci, &c) in col_coords.iter().enumerate() {
                    core_mut.data[r * 4 + c] = val.data()[ri * 2 + ci];
                }
            }
        }
        assert_eq!(t.data()[0], 1.0);
        assert_eq!(t.data()[2], 2.0);
    }
}
