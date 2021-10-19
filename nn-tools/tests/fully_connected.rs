use nntools::{Arr, logic::networks::fully_connected_net};
use std::{cell::RefCell};

use nntools::{data::datasets::mnist::loader::Loader, logic::{activations_fns::{sigmoid}, gradient_decents::{stochastic}, layers::{base_layer::Layer, fully_connected}, loss_fns::{quadratic}}};
use ndarray_rand::{RandomExt, rand_distr::{Normal}};


#[test]
fn success_in_running_fully_connected_nn() {
    let (trn_size, tst_size, val_size, rows, cols) = (5000 as usize, 1000 as usize, 64_000 as usize, 28 as usize, 28 as usize);
    let mnist_loader = Loader::new(
        "./src/data/datasets/mnist/files",
        "train-labels.idx1-ubyte",
        "train-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        trn_size as u32,
        tst_size as u32,
        val_size as u32,
    );

    let (
        trn_img, 
        trn_lbl, 
        tst_img, 
        tst_lbl, 
        _, 
        _ 
    ) = mnist_loader.build();

    let epoches = 8;
    let mini_batch_size = 10 as usize;
    let inputs_size = rows*cols as usize;
    let stochastic = stochastic::Init::new(3.,mini_batch_size);
    let sigmoid = sigmoid::Init {};
    let mut w1 = Arr::random((30, rows*cols), Normal::new(0., 0.03).unwrap());
    let mut b1 = Arr::random((30, 1), Normal::new(0., 1.).unwrap());
    let mut layer_one = fully_connected::Init::new(
        &mut w1,
        &mut b1,
        &stochastic
    );

    let mut w2 = Arr::random((10, 30), Normal::new(0., 0.01).unwrap());
    let mut b2 = Arr::random((10, 1), Normal::new(0., 1.).unwrap());
    let mut layer_two = fully_connected::Init::new(
        &mut w2,
        &mut b2,
        &stochastic
    );

    let quadratic = quadratic::Init::new(&sigmoid);


    let layers_vec: Vec<&mut dyn Layer> = vec!(&mut layer_one, &mut layer_two);
    let layers: RefCell<Vec<&mut dyn Layer>> = RefCell::new(layers_vec);

    let mut i = 0;
    let mut prev_accuracy = f64::MIN;
    let mut counter = 0;

    while i < epoches {
        fully_connected_net::Network::new(
            &trn_img,
            &trn_lbl,
            mini_batch_size,
            inputs_size,
            trn_size,
            layers.borrow_mut(),
            &quadratic,
            &sigmoid
        ).run(true);

        let x = fully_connected_net::Network::new(
            &tst_img,
            &tst_lbl,
            tst_size,
            inputs_size,
            tst_size,
            layers.borrow_mut(),
            &quadratic,
            &sigmoid
        ).run(false);


        println!("accuarcy: {}", 100.*x);
        if x < prev_accuracy {
            counter+=1;
        } else {
            counter = 0;
        }

        prev_accuracy = x;
        assert!(counter <= 1);
        assert!(x > 0.9 || i <= 2);
        i+=1;
   }
}