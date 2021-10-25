use ndarray_rand::rand_distr::Normal;
use nntools::logic::gradient_decents::adagrad;
use nntools::logic::networks::rnn_net::NetworkRunParams;
use nntools::logic::utils::arr_zeros_with_shape;
use nntools::logic::utils::gradient_clipping;
use nntools::{Arr};
use core::cell::RefCell;
use nntools::{data::datasets::warandpeace::loader::{Loader}, logic::{activations_fns::tanh, networks::rnn_net::{self}}};
use ndarray_rand::{RandomExt};
use nntools::logic::{loss_fns::cross_entropy, utils::one_hot_encoding};

/* TODOs:
    arr.dot has performence issues
*/

#[test]
fn success_in_running_rnn() {
    let mini_batch_size = 10;
    let sequence_size = 50;
    let epoches = 7;

    let loader = Loader::new(
        "./src/data/datasets/warandpeace/files/shortend.txt",
        75,
        10,
        15,
        mini_batch_size,
        sequence_size
    );

    let (
        trn_data, 
        trn_lbls,
        _tst_data,
        _tst_lbls,
        _val_data,
        _val_lbls,
        word_dim
    ) = loader.build();


    let new_mini_batch_size = trn_data.shape()[1];
    let bptt_truncate = 4;
    let hidden_dim = 100;
    let tanh = tanh::Init {};
    let learning_rate = 1e-1;
    let adagrad = adagrad::Init::new(learning_rate, mini_batch_size);

    let mut inputs_weights: Arr = Arr::random(
        (hidden_dim, word_dim), 
    Normal::new(0., 0.01).unwrap()
    );
    let mut state_weights = Arr::random(
        (hidden_dim, hidden_dim), 
    Normal::new(0., 0.01).unwrap()
    );
    let mut output_weights = Arr::random(
        (word_dim, hidden_dim), 
    Normal::new(0., 0.01).unwrap()
    );
    let mut state_biases = arr_zeros_with_shape(&[hidden_dim, 1]);
    let mut output_biases = arr_zeros_with_shape(&[word_dim, 1]);

    let f64_trn_data = trn_data.map(|x| *x as f64);
    let f64_trn_lbs = trn_lbls.map(|x| *x as f64);
    let encoded_trn_data = one_hot_encoding(&f64_trn_data, word_dim);
    let cross_entropy = cross_entropy::Init {};

    let rnn_net = rnn_net::Network::new(
        &encoded_trn_data,
        &f64_trn_lbs,
        new_mini_batch_size,
        sequence_size,
        bptt_truncate,
        word_dim,
        hidden_dim,
        &tanh,
        &cross_entropy
    );

    let mut mem_iw = arr_zeros_with_shape(inputs_weights.shape());
    let mut mem_sw = arr_zeros_with_shape(state_weights.shape());
    let mut mem_ow = arr_zeros_with_shape(output_weights.shape());
    let mut mem_sb = arr_zeros_with_shape(state_biases.shape());
    let mut mem_ob = arr_zeros_with_shape(output_biases.shape());

    let output_weights_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut output_weights);
    let input_weights_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut inputs_weights);
    let state_weights_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut state_weights);
    let state_biases_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut state_biases);
    let output_biases_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut output_biases);

    let mut smoth_loss = 
        Arr::from_shape_fn((1, new_mini_batch_size), |(_, _)| -(1. / word_dim as f64).log(2.) * sequence_size as f64);
    
    let mut i = 0;
    let mut prev_loss = f64::MAX;
    let mut counter = 0;

    while i < epoches {
        let net_run_params = NetworkRunParams::new(
            &input_weights_ref_cell,
            &state_weights_ref_cell,
            &output_weights_ref_cell,
            &state_biases_ref_cell,
            &output_biases_ref_cell,
        );

        let min_clipping_value = -5.;
        let max_clipping_value = 5.;
        rnn_net.run(&mut smoth_loss, |(
            du, 
            dw, 
            dv, 
            dbs, 
            dbo
        )| {
            let clipped_du = gradient_clipping(&du, min_clipping_value, max_clipping_value);
            let clipped_dw = gradient_clipping(&dw, min_clipping_value, max_clipping_value);
            let clipped_dv = gradient_clipping(&dv, min_clipping_value, max_clipping_value);
            let cilpped_dbs = gradient_clipping(&dbs, min_clipping_value, max_clipping_value);
            let cilpped_dbo = gradient_clipping(&dbo, min_clipping_value, max_clipping_value);

            adagrad.change_weights(&mut input_weights_ref_cell.borrow_mut(), &clipped_du, &mut mem_iw);
            adagrad.change_weights(&mut state_weights_ref_cell.borrow_mut(), &clipped_dw, &mut mem_sw);
            adagrad.change_weights(&mut output_weights_ref_cell.borrow_mut(), &clipped_dv, &mut mem_ow);
            adagrad.change_biases(&mut state_biases_ref_cell.borrow_mut(), &cilpped_dbs, &mut mem_sb);
            adagrad.change_biases(&mut output_biases_ref_cell.borrow_mut(), &cilpped_dbo, &mut mem_ob);
        }, net_run_params);

        println!("smoth loss: {:?}", smoth_loss);

        let loss = smoth_loss.sum();
        if loss > prev_loss {
            counter+=1;
        } else {
            counter = 0;
        }

        prev_loss = loss;
        assert!(counter <= 1);
        assert!(smoth_loss[(0,0)] < 297. || i <= 2);
        i+=1;
    }
}