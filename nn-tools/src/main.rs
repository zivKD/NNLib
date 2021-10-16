pub mod logic;
pub mod data;

use std::cmp::min;

use ndarray::{Array2, ArrayView2, s};
pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
pub fn DEFAULT() -> Arr { Arr::default((1,1)) }
use logic::{activations_fns::{base_activation_fn::ActivationFN, tanh}, gradient_decents::base_gradient_decent::GradientDecent, layers::base_layer::Layer, loss_fns::base_loss_fn::LossFN, networks::rnn};

use data::datasets::warandpeace::loader::{self, Loader};
use logic::{activations_fns::sigmoid, gradient_decents::stochastic, layers::fully_connected, loss_fns::quadratic};
use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform}};

use crate::logic::{loss_fns::cross_entropy, utils::one_hot_encoding};

/* TODOs: 
    1. loader should return in usize not in f64
    2. functions with performence issues:
        - decouple rnn from softmax and cross_entropy
        - cross_entroy_loss_with_softmax
        - arr.dot which affects feedforward and propogation
*/
fn main() {
    let mini_batch_size = 2;
    let sequence_size = 2;
    let loader = Loader::new(
        "./src/data/datasets/warandpeace/files/tst.txt",
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
        val_data,
        val_lbls,
        word_dim
    ) = loader.build();


    let new_mini_batch_size = trn_data.shape()[1];
    let bptt_truncate = 5;
    let hidden_dim = 100;
    let tanh = tanh::Init {};
    let stochastic = stochastic::Init::new(0.005, mini_batch_size);
    let learning_rate = 0.005;

    let mut inputs_weights: Arr = Arr::random(
        (hidden_dim, word_dim), 
        Normal::new(0., learning_rate).unwrap()
    );
    let mut state_weights = Arr::random(
        (hidden_dim, hidden_dim), 
        Normal::new(0., learning_rate).unwrap()
    );
    let mut output_weights = Arr::random(
        (word_dim, hidden_dim), 
        Normal::new(0., learning_rate).unwrap()
    );

    // Uniform::new(-f64::sqrt(-(1./hidden_dim as f64)), f64::sqrt(1./hidden_dim as f64))

    println!("data shape: {:?}", trn_data.shape());
    println!("lbls shape: {:?}", trn_lbls.shape());
    println!("mini batch size: {}", new_mini_batch_size);
    println!("word dim: {}", word_dim);
    let encoded_trn_data = one_hot_encoding(&trn_data, word_dim);
    // let encoded_trn_lbls = one_hot_encoding(&trn_lbls, word_dim);
    let cross_entropy = cross_entropy::Init {};
    let mut rnn_network = rnn::Network::new(
        &encoded_trn_data,
        &trn_lbls,
        new_mini_batch_size,
        sequence_size,
        bptt_truncate,
        word_dim,
        hidden_dim,
        &stochastic,
        &tanh,
        &mut inputs_weights,
        &mut state_weights,
        &mut output_weights,
        &cross_entropy
    );

    println!("encoded train data shape: {:?}", encoded_trn_data.shape());
    println!("trn lbls shape: {:?}", trn_lbls.shape());
    let mut i = 0;
    while i < 30 {
        rnn_network.run(true);
    }
}
