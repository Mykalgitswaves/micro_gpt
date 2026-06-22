struct Tensor {
    data: Vec<f32>, 
    // The dimensions, e.g., [2, 3] for a 2x3 matrix
    shape: Vec<usize>, 
    // How many elements to skip in the flat Vec to move down a dimension
    strides: Vec<usize>, 
    // Optional gradient for backpropagation
    grad: Option<Box<Tensor>>, 
}