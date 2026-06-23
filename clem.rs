struct Tensor {
    data: Vec<f32>,
}

impl From<f32> for Tensor {
    fn from(x: f32) -> Self {
        Tensor { data: vec![x] }
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(v: Vec<f32>) -> Self {
        Tensor { data: v }
    }
}

fn tensor<T: Into<Tensor>>(data: T) -> Tensor {
    data.into()
}

pub mod clem {
    pub struct Tensor {
        data: Vec<f32>,
        shape: Vec<usize>,
    }

    impl Tensor {
        // Create a tensor from a scalar
        pub fn scalar(x: f32) -> Self {
            Tensor {
                data: vec![x],
                shape: vec![],
            }
        }

        // Create a tensor from a vector
        pub fn vector(data: Vec<f32>) -> Self {
            Tensor {
                shape: vec![data.len()],
                data,
            }
        }

        // Exponential
        pub fn exp(&self) -> Self {
            Tensor {
                data: self.data.iter().map(|x| x.exp()).collect(),
                shape: self.shape.clone(),
            }
        }

        pub fn float(&self) -> Self {
            Tensor {
                data: self.data.clone() as Vec<i8>,
                shape: self.shape.clone()
            }
        }

        // Reshape
        pub fn reshape(&mut self, shape: Vec<usize>) {
            self.shape = shape;
        }
    }
    pub fn masked(tensor: Tensor, mask: fn) -> Tensor {
        return tensor.data.clone().iter().map(|x| mask(x)).collect()
    }
}

impl From<Clem.Tensor> for Tensor {
    // @ is the dot product matrix multiplication needed by 
    // Deep learning libraries
    fn @ () -> {
        // Assert if shape does not match then we throw exception. 

        // If shape matches dot product takes rows and cols of a tensor
        // multiplies them so that each col is multiplied by each row. 
        // e.g.: 
        // [[1,2], [2,3]] @ [[2,2], [4,4]] = 1*2 + 
        // |
        // 1 2 @ 2 4 = [(1 * 2) + (2 * 2), 
        // 2 3   2 4   [ ()
    }
}