mod logic;
mod data;

use logic::base::LossFN as LossFN;

fn main() {
    let a = vec!(4.0, 10.0, 10.0);
    let b = vec!(8.0, 8.0, 8.0);
    let quadratic = logic::loss_fns::QuadraticLossFN {};
    println!("vector after qaudratic {:?}", quadratic.loss(&a, &b))
}