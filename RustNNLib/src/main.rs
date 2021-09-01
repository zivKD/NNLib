// Mods
mod logic;
mod data;
//

// Imports
use logic::activations_fns::sigmoid;
use logic::gradient_decents::stochastic;
use logic::layers::base_layer::Layer;
use logic::layers::fully_connected;
use ndarray::{ Array2, arr2 };
//

// Types
pub type Arr = Array2<f64>;
//

fn main() {
    let sigmoid_activation_fn = sigmoid::init {};
    let weights = &mut arr2(&[[1.,2.], [3.,4.]]);
    let biases = &mut arr2(&[[1.,2.], [4.,5.]]);
    let stochastic = stochastic::init {};
    let mut fully_connected = fully_connected::init::new(
        10,
        10,
        weights,
        biases,
        &sigmoid_activation_fn,
        &stochastic,
    );
    let outputs = fully_connected.feedforward(arr2(&[[2.,3.], [5.,6.]]));
    let error = fully_connected.propogate(arr2(&[[2.,3.], [4.,6.]]));

    println!("outputs: {:?}", outputs);
    println!("error: {:?}", error);
}