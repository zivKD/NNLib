pub mod logic;
pub mod data;
use std::{cell::RefCell};

use ndarray::{Array2, ArrayView2};
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
use ndarray_rand::{RandomExt, rand_distr::{Uniform}};

/*
TODO:
    1. Why is it stuck at 9.9%
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

    // weights = np.random.normal(loc=0, scale=np.sqrt(1/n_out), size=(n_in, n_out))
    // biases = np.random.normal(loc=0, scale=1, size = (n_out,))
    let epoches = 30;
    let mini_batch_size = 10 as usize;
    let inputs_size = rows*cols as usize;
    let stochastic = stochastic::Init::new(3.,mini_batch_size);
    let sigmoid = sigmoid::Init {};
    let mut w1 = Arr::random((30, rows*cols), Uniform::new(0., 0.03));
    let mut b1 = Arr::random((30, 1), Uniform::new(0., 1.));
    let mut layer_one = fully_connected::Init::new(
        &mut w1,
        &mut b1,
        &sigmoid,
        &stochastic
    );

    let mut w2 = Arr::random((10, 30), Uniform::new(0., 0.01));
    let mut b2 = Arr::random((10, 1), Uniform::new(0., 1.));
    let mut layer_two = fully_connected::Init::new(
        &mut w2,
        &mut b2,
        &sigmoid,
        &stochastic
    );

    let quadratic = quadratic::Init {};


    let layers_vec: Vec<&mut dyn Layer> = vec!(&mut layer_one, &mut layer_two);
    let layers: RefCell<Vec<&mut dyn Layer>> = RefCell::new(layers_vec);

    let mut i = 0;
    while i < epoches {
        network::Network::new(
            &trn_img,
            &trn_lbl,
            mini_batch_size,
            inputs_size,
            trn_size,
            layers.borrow_mut(),
            &quadratic
        ).run(false);

        network::Network::new(
            &val_img,
            &val_lbl,
            val_size,
            inputs_size,
            val_size,
            layers.borrow_mut(),
            &quadratic
        ).run(true);

        i+=1;
   }
}
