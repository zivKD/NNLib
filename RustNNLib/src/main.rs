mod logic;
mod data;

// use logic::base::LossFN as LossFN;
use logic::activations_fns::base_activation_fn::ActivationFN;
use logic::activations_fns::relu;
use logic::loss_fns::base_loss_fn::LossFN;
use logic::loss_fns::quadratic;
use ndarray::Array;
use ndarray::{ Array2 };

pub type Arr = Array2<f64>;

fn main() {
    // let relu = relu::init {};
    // let mut z1 = Array::from_elem((2, 2), 1.);
    // relu.forward(&mut z1);
    // println!("sig forward {:?}", z1);
    // let mut z2 = Array::from_elem((2, 2), 1.);
    // relu.propogate(&mut z2);
    // println!("sig propogate {:?}", z2);
    let quadratic = quadratic::init {};
    let mut z1 = Array::from_elem((2, 2), 4.);
    let z2 = Array::from_elem((2, 2), 1.);
    let result1 = quadratic.output(&mut z1, &z2);
    println!("quadratic output {:?}", result1);
    let mut z3 = Array::from_elem((2, 2), 4.);
    let z4 = Array::from_elem((2, 2), 1.);
    let result2 = quadratic.propogate(&mut z3, &z4);
    println!("quadratic propgate {:?}", result2);
}