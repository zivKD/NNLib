use std::iter::FromIterator;

use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::{Arr, ArrViewMut};
use ndarray::{Array1, ArrayBase, Axis, Zip};
use crate::logic::layers::base_layer::Layer;

pub struct init<'a> {
    n_in: usize,
    n_out: usize,
    weights: &'a mut Arr,
    biases: &'a mut Arr,
    activation_fn: &'a dyn ActivationFN,
    gradient_decent: &'a dyn GradientDecent,
    // current_inputs: Arr,
    current_weighted_inputs: Arr,
    current_activations: Arr,
}

impl init<'_> {
    pub fn new<'a>(
        n_in: usize, 
        n_out: usize, 
        weights: &'a mut Arr, 
        biases: &'a mut Arr, 
        activation_fn: &'a dyn ActivationFN, 
        gradient_decent: &'a dyn GradientDecent
    ) -> init<'a> {
        init {
            activation_fn,
            n_in,
            n_out,
            weights,
            biases,
            gradient_decent,
            current_weighted_inputs: Arr::default((n_in, n_out)),
            current_activations: Arr::default((n_in, n_out))
        }
    }
}

impl Layer for init<'_> {
    fn feedforward(&mut self, inputs: ArrViewMut) -> Arr {
        let mut dots = inputs.dot(self.weights);
        println!("got here 3a");
        // Probably not cost affective, should improve performenece
        let mut repeated_biases = Arr::from_shape_fn((dots.shape()[0], dots.shape()[1]), 
                                                                            |(_i, j)| *self.biases.get((j, 0)).unwrap()
                                                                            );
        Zip::from(&mut dots).and(&mut repeated_biases).for_each(|dot, &mut bias| *dot += bias);
        println!("got here 3b");
        self.current_weighted_inputs = dots.clone();
        println!("got here 3c");
        let a = self.activation_fn.forward(&dots);
        println!("got here 3d");
        self.current_activations = a.clone();
        println!("got here 3e");
        a
    }

    fn propogate(&mut self, gradient: Arr) -> Arr {
        let w_gradient : Arr = gradient.dot(&self.current_activations.t());
        self.gradient_decent.change_weights(self.weights, &w_gradient);
        self.gradient_decent.change_biases(self.biases, &gradient);
        let mut activation_derivative = self.activation_fn.propogate(&mut self.current_weighted_inputs);
        let weighted_error: Arr = self.weights.dot(&gradient);
        Zip::from(&mut activation_derivative).and(&weighted_error).for_each(|x, &y| *x *= y);
        activation_derivative
    }
}