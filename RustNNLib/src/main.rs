mod logic;
mod data;

// use logic::base::LossFN as LossFN;
use logic::activations_fns::base_activation_fn::activation_fn;
use logic::activations_fns::relu;
use ndarray::Array;

fn main() {
    let relu = relu::init {};
    let mut z1 = Array::from_elem((2, 2), 1.);
    relu.forward(&mut z1);
    println!("sig forward {:?}", z1);
    let mut z2 = Array::from_elem((2, 2), 1.);
    relu.propogate(&mut z2);
    println!("sig propogate {:?}", z2);
}