pub mod logic;
pub mod data;

use ndarray::{Array2, ArrayView2, s};
pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
pub fn DEFAULT() -> Arr { Arr::default((1,1)) }
use logic::{activations_fns::{base_activation_fn::ActivationFN, tanh}, gradient_decents::base_gradient_decent::GradientDecent, layers::base_layer::Layer, loss_fns::base_loss_fn::LossFN, networks::rnn};

use data::datasets::warandpeace::loader::{self, Loader};
use logic::{activations_fns::sigmoid, gradient_decents::stochastic, layers::fully_connected, loss_fns::quadratic};
use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform}};


fn main() {
    let mini_batch_size = 10;
    let sequence_size = 50;
    let loader = Loader::new(
        "./src/data/datasets/warandpeace/files/sources.txt",
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

    let mut rnn_network = rnn::Network::new(
        &trn_data,
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
        &mut output_weights
    );

    println!("worddim {}", word_dim);
    // println!("trn data shape: {:?}", trn_data.shape());
    // println!("trn lbls shape: {:?}", trn_lbls.shape());

    let mut i = 0;
    while i < 30 {
        rnn_network.run(false);
    }
}
