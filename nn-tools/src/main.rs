pub mod logic;
pub mod data;
use std::cmp::min;

use ndarray::{Array, Array2, ArrayBase, ArrayView2, ArrayViewMut, ArrayViewMut2, Axis, Slice, s};
pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
use ndarray_stats::QuantileExt;

use logic::{
    activations_fns::base_activation_fn::ActivationFN, 
    gradient_decents::base_gradient_decent::GradientDecent, 
    layers::base_layer::Layer, 
    loss_fns::base_loss_fn::LossFN
};
use logic::{activations_fns::sigmoid, gradient_decents::stochastic, layers::fully_connected, loss_fns::quadratic};
use ndarray::{ arr2 };
use mnist::{Mnist,MnistBuilder};

use crate::logic::utils::{repeat, repeated_axis_zero};


/*
TODO:
    1. How to optimize the library, when to use borrowing, when to copy and so on...
    2. Create a marco network!(layers... inputs... labels... mini_batch_size... epoches_num... loss_fn...)
*/

fn main() {
    let (trn_size, tst_size, val_size, rows, cols) = (50_000 as usize, 10_000 as usize, 10_000 as usize, 28 as usize, 28 as usize);
    // Deconstruct the returned Mnist struct.
    let mnist = MnistBuilder::new()
        .base_path("./src/datasets/MNIST")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .validation_set_length(val_size as u32)
        .test_set_length(tst_size as u32)
        .finalize();
    
    let mut trn_img_f64 : Vec<f64> = mnist.trn_img.iter().map(|x| *x as f64).collect();
    let mut trn_lbl_f64 : Vec<f64> = mnist.trn_lbl.iter().map(|x| *x as f64).collect();
    let trn_img = Arr::from_shape_vec((trn_size*rows*cols, 1), trn_img_f64).unwrap();
    let trn_lbl = Arr::from_shape_vec((trn_size, 1), trn_lbl_f64).unwrap();

    // let mut val_img_f64 : Vec<f64> = mnist.val_img.iter().map(|x| *x as f64).collect();
    // let mut val_lbl_f64 : Vec<f64> = mnist.val_lbl.iter().map(|x| *x as f64).collect();
    // let val_img = Arr::from_shape_vec(((val_size*rows) as usize, cols as usize), val_img_f64);
    // let val_lbl = Arr::from_shape_vec(((val_size*rows) as usize, cols as usize), val_lbl_f64);

    // let mut tst_img_f64 : Vec<f64> = mnist.tst_img.iter().map(|x| *x as f64).collect();
    // let mut tst_lbl_f64 : Vec<f64> = mnist.tst_lbl.iter().map(|x| *x as f64).collect();
    // let tst_img = Arr::from_shape_vec(((tst_size*rows) as usize, cols as usize), tst_img_f64);
    // let tst_lbl = Arr::from_shape_vec(((tst_size*rows) as usize, cols as usize), tst_lbl_f64);

    let epoches = 1;
    let mini_batch_size = 10 as usize;
    let inputs_size = rows*cols as usize;
    let stochastic = stochastic::init::new(0.03,mini_batch_size);
    let sigmoid = sigmoid::init {};
    let mut w1 = Arr::zeros((30, rows*cols));
    let mut b1 = Arr::zeros((30, 1));
    let mut layer_one = fully_connected::init::new(
        784,
        30,
        &mut w1,
        &mut b1,
        &sigmoid,
        &stochastic
    );

    let mut w2 = Arr::zeros((10, 30));
    let mut b2 = Arr::zeros((10, 1));
    let mut layer_two = fully_connected::init::new(
       30,
        10,
        &mut w2,
        &mut b2,
        &sigmoid,
        &stochastic
    );

    let quadratic = quadratic::init {};


    let mut i = 0;
    while i < epoches {
        let mut iteration = 1;
        let mut lower_bound = 0;
        let mut higher_bound = mini_batch_size * inputs_size;
        while higher_bound < trn_size*inputs_size {
            let mini_batch: ArrView = trn_img.slice(s![lower_bound..higher_bound, ..]);
            let mini_batch = mini_batch
                                                            .into_shape((inputs_size, mini_batch_size)).unwrap();
            let mini_batch_lbs = trn_lbl.slice(s![(iteration-1)*mini_batch_size..iteration*mini_batch_size, ..]).to_owned();
             /*
                Return a 10-dimensional unit vector with a 1.0 in the jth
                position and zeroes elsewhere.  This is used to convert a digit
                (0...9) into a corresponding desired output from the neural
                network.
            */
            let mini_batch_lbs = Arr::from_shape_fn((mini_batch_size, 10), |(i, j)| {
                let mut val = 0.;
                if *mini_batch_lbs.get((i, 0)).unwrap() as usize == j {
                    val = 1.;
                }

                val
            });
            let inputs = layer_one.feedforward(mini_batch);
            let view_inputs = inputs.view(); 
            let mut outputs = layer_two.feedforward(view_inputs); 
            let error = quadratic.propogate(&mut outputs, &mini_batch_lbs) ;
            let next_error = layer_two.propogate(error,view_inputs);
            layer_one.propogate(next_error, mini_batch);
            iteration+=1;
            lower_bound = (iteration - 1) * (mini_batch_size * inputs_size);
            higher_bound = iteration * mini_batch_size * inputs_size;
        }

        // let mini_batch = train_set.slice_mut(s![0..10, ..]);
        // let mut inputs = layer_one.feedforward(mini_batch);
        // let view_inputs = inputs.view_mut(); 
        // let mut outputs = layer_two.feedforward(view_inputs); 
        i+=1;
   }
}
