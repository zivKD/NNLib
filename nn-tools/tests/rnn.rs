use ndarray::Slice;
use ndarray::Array2;
use ndarray::Order;
use ndarray::Zip;
use ndarray::Axis;
use ndarray::s;
use ndarray_rand::rand_distr::Normal;
use ndarray_stats::QuantileExt;
use nntools::ArrView;
use nntools::logic::gradient_decents::adagrad;
use nntools::logic::networks::rnn_net::NetworkRunParams;
use nntools::logic::utils::arr_zeros_with_shape;
use nntools::logic::utils::gradient_clipping;
use nntools::{Arr, logic::layers::rnn_step};
use core::cell::RefCell;
use nntools::{data::datasets::warandpeace::loader::{Loader}, logic::{activations_fns::tanh, gradient_decents::stochastic, networks::rnn_net::{self}}};
use ndarray_rand::{RandomExt, rand_distr::{Uniform}};
use nntools::logic::{loss_fns::cross_entropy, utils::one_hot_encoding};

/* TODOs:
    1. loader should return in usize not in f64
    2. functions with performence issues:
        - decouple rnn from softmax and cross_entropy
        - cross_entroy_loss_with_softmax
        - arr.dot which affects feedforward and propogation
*/

#[test]
fn success_in_running_rnn() {
    let mini_batch_size = 10;
    let sequence_size = 25;
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
        tst_data,
        tst_lbls,
        _val_data,
        _val_lbls,
        word_dim
    ) = loader.build();


    let new_mini_batch_size = trn_data.shape()[1];
    let bptt_truncate = 4;
    let hidden_dim = 100;
    let tanh = tanh::Init {};
    let learning_rate = 1e-1;
    let stochastic = stochastic::Init::new(learning_rate, mini_batch_size);
    let adagrad = adagrad::Init::new(learning_rate, mini_batch_size);
    let word_dim_limit = (1./word_dim as f64).sqrt();
    let hidden_dim_limit = (1./hidden_dim as f64).sqrt();

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

    let encoded_trn_data = one_hot_encoding(&trn_data, word_dim);
    let encoded_tst_data = one_hot_encoding(&tst_data, word_dim);
    let cross_entropy = cross_entropy::Init {};

    let rnn_net = rnn_net::Network::new(
        &encoded_trn_data,
        &trn_lbls,
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
    while i < 30 {
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

        // let trn_data_size = encoded_trn_data.shape()[0];
        // while higher_bound <= (trn_data_size - size) {
        //     let mini_batch: ArrView = encoded_trn_data.slice(s![lower_bound..higher_bound, ..]);
        //     let mini_batch = mini_batch
        //                                                     .to_shape(((size, new_mini_batch_size), Order::ColumnMajor)).unwrap().to_owned();
        //     let mini_batch_lbs = trn_lbls.slice_axis(
        //         Axis(0), 
        //         Slice::from((iteration-1)*sequence_size..iteration*sequence_size)
        //     ).to_owned();

        //     let (du, dw, dv, dbs, dbo) = 
        //         rnn_unit.run_single_step(&mini_batch, &mini_batch_lbs, &mut smoth_loss);
        //     iteration+=1;
        //     lower_bound = (iteration - 1) * size;
        //     higher_bound = iteration * size;
        // }

        println!("smoth loss: {:?}", smoth_loss);

        // let sw_ref = state_weights_ref_cell.borrow();
        // let iw_ref = input_weights_ref_cell.borrow();
        // let ow_ref = output_weights_ref_cell.borrow();
        // let sb_ref = state_biases_ref_cell.borrow();
        // let ob_ref = output_biases_ref_cell.borrow();

        // let mut rnn_unit = rnn_step::Init::new(
        //     &sw_ref,
        //     &iw_ref,
        //     &ow_ref,
        //     &sb_ref,
        //     &ob_ref,
        //     &tanh,
        // );

        // let first_batch_encoded = encoded_tst_data.slice(s![0..word_dim, ..]);
        // let first_batch_encoded = first_batch_encoded.to_shape(((word_dim, new_mini_batch_size), Order::ColumnMajor)).unwrap().to_owned();
        // let prev_s = arr_zeros_with_shape(&[hidden_dim, new_mini_batch_size]);
        // rnn_unit.feedforward((&first_batch_encoded.view(), prev_s.view()));
        // let softmaxed_output = cross_entropy.softmax_forward(&rnn_unit.mulv);
        // let first_batch = tst_data.column(0).slice(s![0..new_mini_batch_size]).to_shape((1, new_mini_batch_size)).unwrap().to_owned();;
        // let first_batch_lbs = tst_lbls.column(0).slice(s![0..new_mini_batch_size]).to_shape((1, new_mini_batch_size)).unwrap().to_owned();;
        // let predictions: Array2<usize> = softmaxed_output.
        //     map_axis(Axis(0), |a| a.argmax().unwrap()).
        //     to_shape((1, new_mini_batch_size)).unwrap().to_owned();
        // let mut accuracy = 0;
        // Zip::from(&predictions).and(&first_batch_lbs).and(&first_batch).for_each(|p, l, i| {
        //     println!("input value: {}, actual value: {}, predicted value: {}",*i, *l, *p);
        //     if *p as f64 == *l {
        //         accuracy+=1;
        //     }
        // });
        
        // println!("accuracy {}", accuracy);
        i+=1;
    }
}