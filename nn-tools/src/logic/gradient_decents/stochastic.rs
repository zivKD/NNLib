use crate::Arr;
use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use ndarray::Zip;

pub struct init {
    learning_rate: f64,
    mini_batch_size: i32
}

impl init {
    pub fn new(learning_rate: f64, mini_batch_size: i32) -> Self {
        init {
            learning_rate,
            mini_batch_size
        }
    }
}

impl GradientDecent for init {
    fn change_weights<'a>(&self, w: &'a mut Arr, gradient: & Arr) {
        let rate = self.learning_rate / self.mini_batch_size as f64;
        let balanced_gradient = rate * gradient;
        Zip::from(w).and(&balanced_gradient).for_each(|x, &y| *x -= y);
    }

    fn change_biases<'a>(&self, b: &'a mut Arr, gradient: & Arr) {
        let rate = self.learning_rate / self.mini_batch_size as f64;
        let balanced_gradient = rate * gradient;
        Zip::from(b).and(&balanced_gradient).for_each(|x, &y| *x -= y);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use ndarray::arr2;
    use crate::logic::utils::round_decimal;

    #[test]
    fn stochastic_biases_change(){
        let stochastic: init = init::new(0.03, 10);
        let gradient : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
        let mut biases : Arr = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        stochastic.change_biases(&mut biases, &gradient);
        let expected_result = arr2(&[[0.097, 0.198404, 0.297558], [0.3990691, 0.4986956, 0.59964]]);
        assert_eq!(biases.mapv(|x| round_decimal(7, x)), expected_result);
    }

    #[test]
    fn stochastic_weights_change() {
        let stochastic: init = init::new(0.03, 10);
        let gradient : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
        let mut weights : Arr = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        stochastic.change_weights(&mut weights, &gradient);
        let expected_result = arr2(&[[0.097, 0.198404, 0.297558], [0.3990691, 0.4986956, 0.59964]]);
        assert_eq!(weights.mapv(|x| round_decimal(7, x)), expected_result);
    }
}