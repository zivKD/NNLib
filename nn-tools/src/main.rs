pub mod logic;
pub mod data;
use ndarray::Array2;
pub type Arr = Array2<f64>;

use mnist::{Mnist,MnistBuilder};

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
    let trn_img = Arr::from_shape_vec(((trn_size*rows) as usize, cols as usize), trn_img_f64);
    let trn_lbl = Arr::from_shape_vec(((trn_size*rows) as usize, cols as usize), trn_lbl_f64);

    let mut val_img_f64 : Vec<f64> = mnist.val_img.iter().map(|x| *x as f64).collect();
    let mut val_lbl_f64 : Vec<f64> = mnist.val_lbl.iter().map(|x| *x as f64).collect();
    let val_img = Arr::from_shape_vec(((val_size*rows) as usize, cols as usize), val_img_f64);
    let val_lbl = Arr::from_shape_vec(((val_size*rows) as usize, cols as usize), val_lbl_f64);

    let mut tst_img_f64 : Vec<f64> = mnist.tst_img.iter().map(|x| *x as f64).collect();
    let mut tst_lbl_f64 : Vec<f64> = mnist.tst_lbl.iter().map(|x| *x as f64).collect();
    let tst_img = Arr::from_shape_vec(((tst_size*rows) as usize, cols as usize), tst_img_f64);
    let tst_lbl = Arr::from_shape_vec(((tst_size*rows) as usize, cols as usize), tst_lbl_f64);

}
