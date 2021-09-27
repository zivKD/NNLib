pub mod logic;
pub mod data;
use ndarray::{Array2, ArrayView2, s};
pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
use logic::{
    activations_fns::base_activation_fn::ActivationFN, 
    gradient_decents::base_gradient_decent::GradientDecent, 
    layers::base_layer::Layer, 
    loss_fns::base_loss_fn::LossFN,
    network
};
use data::datasets::mnist::loader::Loader;
use logic::{activations_fns::sigmoid, gradient_decents::stochastic, layers::fully_connected, loss_fns::quadratic};

/*
TODO:
    1. How to optimize the library, when to use borrowing, when to copy and so on...
*/

fn main() {
    let (trn_size, tst_size, val_size, rows, cols) = (50_000 as usize, 10_000 as usize, 10_000 as usize, 28 as usize, 28 as usize);
    // Deconstruct the returned Mnist struct.
    let mnist_loader = Loader::new(
        "./src/data/datasets/mnist/files",
        "train-labels.idx1-ubyte",
        "train-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        trn_size as u32,
        tst_size as u32,
        val_size as u32,
        rows,
        cols
    );

    let (
        trn_img, 
        trn_lbl, 
        tst_img, 
        tst_lbl, 
        val_img, 
        val_lbl
    ) = mnist_loader.build();

    let epoches = 1;
    let mini_batch_size = 10 as usize;
    let inputs_size = rows*cols as usize;
    let stochastic = stochastic::Init::new(0.03,mini_batch_size);
    let sigmoid = sigmoid::Init {};
    let mut w1 = Arr::zeros((30, rows*cols));
    let mut b1 = Arr::zeros((30, 1));
    let mut layer_one = fully_connected::Init::new(
        &mut w1,
        &mut b1,
        &sigmoid,
        &stochastic
    );

    let mut w2 = Arr::zeros((10, 30));
    let mut b2 = Arr::zeros((10, 1));
    let mut layer_two = fully_connected::Init::new(
        &mut w2,
        &mut b2,
        &sigmoid,
        &stochastic
    );

    let quadratic = quadratic::Init {};

    let layers: Vec<&mut Layer> = vec!(&mut layer_one, &mut layer_two);
    let mut network = network::Network::new(
        trn_img,
        trn_lbl,
        mini_batch_size,
        inputs_size,
        trn_size,
        layers,
        &quadratic
    );


    let mut i = 0;
    while i < epoches {
        network.run();
        i+=1;
   }
}
