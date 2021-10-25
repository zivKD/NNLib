use crate::{Arr};
use ndarray::Zip;

pub struct Init {
    learning_rate: f64,
    mini_batch_size: usize,
}

impl Init {
    pub fn new(learning_rate: f64, mini_batch_size: usize) -> Self {
        Init {
            learning_rate,
            mini_batch_size,
        }
    }

    pub fn change_weights<'a>(&self, w: &'a mut Arr, gradient: &Arr, mem: &mut Arr) {
        let rate = self.learning_rate / self.mini_batch_size as f64;
        Zip::from(gradient).and(mem).and(w).for_each(|x, y, w| {
            *y += x*x;
            *w -= (rate * x) / ((*y + 1e-8).sqrt());
        });
    }

    pub fn change_biases<'a>(&self, b: &'a mut Arr, gradient: &Arr, mem: &mut Arr) {
        let rate = self.learning_rate / self.mini_batch_size as f64;
        Zip::from(gradient).and(mem).and(b).for_each(|x, y, w| {
            *y += x*x;
            *w -= (rate * x) / ((*y + 1e-8).sqrt());
        });
    }
}


// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::Arr;
//     use ndarray::arr2;
//     use crate::logic::utils::round_decimal;

//     #[test]
//     fn stochastic_biases_change(){
//         let mut adagrad: Init = Init::new(0.03, 10);
//         let gradient : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
//         let mut biases : Arr = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
//         // adagrad.change_biases(&mut biases, &gradient);
//         let expected_result = arr2(&[[0.097, 0.198404, 0.297558], [0.3990691, 0.4986956, 0.59964]]);
//         assert_eq!(biases.mapv(|x| round_decimal(7, x)), expected_result);
//     }

//     #[test]
//     fn stochastic_weights_change() {
//         let mut adagrad: Init = Init::new(0.03, 10);
//         let gradient : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
//         let mut weights : Arr = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
//         // adagrad.change_weights(&mut weights, &gradient);
//         let expected_result = arr2(&[[0.097, 0.198404, 0.297558], [0.3990691, 0.4986956, 0.59964]]);
//         assert_eq!(weights.mapv(|x| round_decimal(7, x)), expected_result);
//     }
// }