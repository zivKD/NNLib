pub mod logic;
pub mod data;
use ndarray::{Array2, ArrayView2, s};
pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
use logic::{
    activations_fns::base_activation_fn::ActivationFN, 
    gradient_decents::base_gradient_decent::GradientDecent, 
    layers::base_layer::Layer, 
    loss_fns::base_loss_fn::LossFN
};
use logic::{activations_fns::sigmoid, gradient_decents::stochastic, layers::fully_connected, loss_fns::quadratic};
use mnist::{MnistBuilder};

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
    
    let trn_img_f64 : Vec<f64> = mnist.trn_img.iter().map(|x| *x as f64).collect();
    let trn_lbl_f64 : Vec<f64> = mnist.trn_lbl.iter().map(|x| *x as f64).collect();
    let trn_img = Arr::from_shape_vec((trn_size*rows*cols, 1), trn_img_f64).unwrap();
    let trn_lbl = Arr::from_shape_vec((trn_size, 1), trn_lbl_f64).unwrap();
    /* Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network.
    */
    let trn_lbl = Arr::from_shape_fn((trn_size, 10), |(i, j)| {
        let mut val = 0.;
        if *trn_lbl.get((i, 0)).unwrap() as usize == j {
            val = 1.;
        }

        val
    });

    // let tst_img_f64 : Vec<f64> = mnist.tst_img.iter().map(|x| *x as f64).collect();
    // let tst_lbl_f64 : Vec<f64> = mnist.tst_lbl.iter().map(|x| *x as f64).collect();
    // let tst_img = Arr::from_shape_vec((tst_size*rows*cols, 1), tst_img_f64).unwrap();
    // let tst_lbl = Arr::from_shape_vec((tst_size, 1), tst_lbl_f64).unwrap();

    let val_img_f64 : Vec<f64> = mnist.val_img.iter().map(|x| *x as f64).collect();
    let val_lbl_f64 : Vec<f64> = mnist.val_lbl.iter().map(|x| *x as f64).collect();
    let val_img = Arr::from_shape_vec((val_size*rows*cols, 1), val_img_f64).unwrap();
    let val_lbl = Arr::from_shape_vec((val_size, 1), val_lbl_f64).unwrap();

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


    let mut i = 0;
    while i < epoches {
        let mut iteration = 1;
        let mut lower_bound = 0;
        let mut higher_bound = mini_batch_size * inputs_size;
        while higher_bound < trn_size*inputs_size {
            println!("running: {}", iteration);
            let mini_batch: ArrView = trn_img.slice(s![lower_bound..higher_bound, ..]);
            let mini_batch = mini_batch
                                                            .into_shape((inputs_size, mini_batch_size)).unwrap();
            let mini_batch_lbs = trn_lbl.slice(s![(iteration-1)*mini_batch_size..iteration*mini_batch_size, ..]).to_owned();
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


        i+=1;
   }
}
