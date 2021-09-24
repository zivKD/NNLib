pub mod logic;
pub mod data;
use ndarray::{Array, Array2, ArrayViewMut, ArrayViewMut2, Axis, Slice, s};
pub type Arr = Array2<f64>;
pub type ArrViewMut<'a> = ArrayViewMut2<'a, f64>;
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


/*
TODO:
    1. How to optimize the library, when to use borrowing, when to copy and so on...
    2. Create a marco network!(layers... inputs... labels... mini_batch_size... epoches_num... loss_fn...)
*/

fn main() {
    let (trn_size, tst_size, val_size, rows, cols) = (50_000, 10_000, 10_000, 28, 28);
    // Deconstruct the returned Mnist struct.
    let mnist = MnistBuilder::new()
        .base_path("./src/datasets/MNIST")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(val_size)
        .test_set_length(tst_size)
        .finalize();
    
    let mut trn_img_f64 : Vec<f64> = mnist.trn_img.iter().map(|x| *x as f64).collect();
    let mut trn_lbl_f64 : Vec<f64> = mnist.trn_lbl.iter().map(|x| *x as f64).collect();
    let trn_img = Arr::from_shape_vec(((trn_size*rows*cols) as usize, 1), trn_img_f64).unwrap();
    let trn_lbl = Arr::from_shape_vec((trn_size as usize, 1 as usize), trn_lbl_f64).unwrap();

    // let mut val_img_f64 : Vec<f64> = mnist.val_img.iter().map(|x| *x as f64).collect();
    // let mut val_lbl_f64 : Vec<f64> = mnist.val_lbl.iter().map(|x| *x as f64).collect();
    // let val_img = Arr::from_shape_vec(((val_size*rows) as usize, cols as usize), val_img_f64);
    // let val_lbl = Arr::from_shape_vec(((val_size*rows) as usize, cols as usize), val_lbl_f64);

    // let mut tst_img_f64 : Vec<f64> = mnist.tst_img.iter().map(|x| *x as f64).collect();
    // let mut tst_lbl_f64 : Vec<f64> = mnist.tst_lbl.iter().map(|x| *x as f64).collect();
    // let tst_img = Arr::from_shape_vec(((tst_size*rows) as usize, cols as usize), tst_img_f64);
    // let tst_lbl = Arr::from_shape_vec(((tst_size*rows) as usize, cols as usize), tst_lbl_f64);

    let epoches = 1;
    let mini_batch_size = 10;
    let inputs_size = rows*cols;
    let stochastic = stochastic::init::new(0.03,mini_batch_size);
    let sigmoid = sigmoid::init {};
    let mut w1 = Arr::zeros((30, (rows*cols) as usize));
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
        let mut lower_bound = 0;
        let mut higher_bound = mini_batch_size * inputs_size as i32;
        let mut train_set = trn_img.clone();
        let train_lbl: Arr = trn_lbl.clone();
        println!("got here 1");
        while higher_bound <= trn_size as i32 {
            let mini_batch = train_set.slice_mut(s![(lower_bound as usize)..(higher_bound as usize), ..])
                                                            .into_shape((inputs_size as usize, mini_batch_size as usize)).unwrap();
            println!("got here 2");
            let mini_batch_lbs = train_lbl.slice(s![(lower_bound as usize)..(mini_batch_size as usize), ..]).to_owned();
            println!("got here 3");
            let mut inputs = layer_one.feedforward(mini_batch);
            println!("got here 4");
            let view_inputs = inputs.view_mut(); 
            println!("got here 5");
            let outputs = layer_two.feedforward(view_inputs); 
            println!("got here 6 {:?}", outputs.shape());
            let mut max_outputs = outputs.map_axis(Axis(1), |x| *x.get( x.argmax().unwrap()).unwrap()).
                                                            into_shape((mini_batch_size as usize, 1)).unwrap();
            println!("got here 7 {:?} {:?}", mini_batch_lbs.shape(), max_outputs.shape());
            let error = quadratic.propogate(&mut max_outputs, &mini_batch_lbs);
            println!("got here 8");
            let next_error = layer_two.propogate(error);
            println!("got here 9");
            layer_one.propogate(next_error);
            println!("got here 10");
            lower_bound+= mini_batch_size;
            higher_bound+= mini_batch_size;
        }

        let mini_batch = train_set.slice_mut(s![0..10, ..]);
        let mut inputs = layer_one.feedforward(mini_batch);
        let view_inputs = inputs.view_mut(); 
        let mut outputs = layer_two.feedforward(view_inputs); 
        println!("{:?}", quadratic.output(&mut outputs, &trn_lbl));
        i+=1;
   }
}
