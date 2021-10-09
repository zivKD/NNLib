pub mod logic;
pub mod data;
use std::{cell::RefCell};

use ndarray::{Array2, ArrayView2, s};
pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
pub fn DEFAULT() -> Arr { Arr::default((1,1)) }
use logic::{
    activations_fns::base_activation_fn::ActivationFN, 
    gradient_decents::base_gradient_decent::GradientDecent, 
    layers::base_layer::Layer, 
    loss_fns::base_loss_fn::LossFN,
};

use data::datasets::warandpeace::loader::Loader;
use logic::{activations_fns::sigmoid, gradient_decents::stochastic, layers::fully_connected, loss_fns::quadratic};
use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform}};


fn main() {
}
