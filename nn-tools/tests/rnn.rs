use ndarray::Array2;
use ndarray::Order;
use ndarray::Zip;
use ndarray::Axis;
use ndarray::s;
use ndarray_stats::QuantileExt;
use nntools::logic::utils::arr_zeros_with_shape;
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
    let sequence_size = 50;
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
    let bptt_truncate = 50;
    let hidden_dim = 100;
    let tanh = tanh::Init {};
    let learning_rate = 0.0001;
    let stochastic = stochastic::Init::new(learning_rate, mini_batch_size);
    let word_dim_limit = (1./word_dim as f64).sqrt();
    let hidden_dim_limit = (1./hidden_dim as f64).sqrt();

    let mut inputs_weights: Arr = Arr::random(
        (hidden_dim, word_dim), 
        Uniform::new(-word_dim_limit, word_dim_limit)
    );
    let mut state_weights = Arr::random(
        (hidden_dim, hidden_dim), 
        Uniform::new(-hidden_dim_limit, hidden_dim_limit)
    );
    let mut output_weights = Arr::random(
        (word_dim, hidden_dim), 
        Uniform::new(-hidden_dim_limit, hidden_dim_limit)
    );

    let encoded_trn_data = one_hot_encoding(&trn_data, word_dim);
    let encoded_tst_data = one_hot_encoding(&tst_data, word_dim);
    let cross_entropy = cross_entropy::Init {};

    let output_weights_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut output_weights);
    let input_weights_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut inputs_weights);
    let state_weights_ref_cell: RefCell<&mut Arr> = RefCell::new(&mut state_weights);

    let mut i = 0;
    while i < 30 {
        rnn_net::Network::new(
            &encoded_trn_data,
            &trn_lbls,
            new_mini_batch_size,
            sequence_size,
            bptt_truncate,
            word_dim,
            hidden_dim,
            &stochastic,
            &tanh,
            input_weights_ref_cell.borrow_mut(),
            state_weights_ref_cell.borrow_mut(),
            output_weights_ref_cell.borrow_mut(),
            &cross_entropy
        ).run(false);

        let sw_ref = state_weights_ref_cell.borrow();
        let iw_ref = input_weights_ref_cell.borrow();
        let ow_ref = output_weights_ref_cell.borrow();
        let mut rnn_unit = rnn_step::Init::new(
            &sw_ref,
            &iw_ref,
            &ow_ref,
            &tanh,
        );

        let first_batch_encoded = encoded_tst_data.slice(s![0..word_dim, ..]);
        let first_batch_encoded = first_batch_encoded.to_shape(((word_dim, new_mini_batch_size), Order::ColumnMajor)).unwrap().to_owned();
        let prev_s = arr_zeros_with_shape(&[hidden_dim, new_mini_batch_size]);
        rnn_unit.feedforward((&first_batch_encoded.view(), prev_s.view()));
        let softmaxed_output = cross_entropy.softmax_forward(&rnn_unit.mulv);
        let first_batch = tst_data.column(0).slice(s![0..new_mini_batch_size]).to_shape((1, new_mini_batch_size)).unwrap().to_owned();;
        let first_batch_lbs = tst_lbls.column(0).slice(s![0..new_mini_batch_size]).to_shape((1, new_mini_batch_size)).unwrap().to_owned();;
        let predictions: Array2<usize> = softmaxed_output.
            map_axis(Axis(0), |a| a.argmax().unwrap()).
            to_shape((1, new_mini_batch_size)).unwrap().to_owned();
        let mut accuracy = 0;
        Zip::from(&predictions).and(&first_batch_lbs).and(&first_batch).for_each(|p, l, i| {
            println!("input value: {}, actual value: {}, predicted value: {}",*i, *l, *p);
            if *p as f64 == *l {
                accuracy+=1;
            }
        });
        
        println!("accuracy {}", accuracy);
        i+=1;
    }
}