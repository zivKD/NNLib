use std::iter::FromIterator;

use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::utils::{repeat, repeated_axis_zero};
use crate::{Arr, ArrView};
use ndarray::{Array1, ArrayBase, Axis, ShapeBuilder, Zip};
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
    fn feedforward(&mut self, inputs: ArrView) -> Arr {
        let mut dots = self.weights.dot(&inputs);
        // Probably not cost affective, should improve performenece
        let mut repeated_biases = repeated_axis_zero(self.biases,&(dots.shape()[0], dots.shape()[1]));
        Zip::from(&mut dots).and(&mut repeated_biases).for_each(|dot, &mut bias| *dot += bias);
        self.current_weighted_inputs = dots.clone();
        let a = self.activation_fn.forward(&dots);
        a
    }

    fn propogate(&mut self, gradient: Arr, activations: ArrView) -> Arr {
        // NOT COMPLETELY SURE THE MATH IS RIGHT
        // println!("δl=δl+1⊙σ′(zl)");
        let gradient =  gradient * self.activation_fn.propogate(&self.current_weighted_inputs);
        // println!("∂C∂wl=δl*wl^T");
        self.gradient_decent.change_weights(self.weights, &gradient.dot(&activations.t()));
        // println!("∂C∂bl=δl");
        // CURENTLY SUM, MABYE BETTER IDEA
        let summed_gradient = gradient.sum_axis(Axis(1)).into_shape((gradient.shape()[0], 1)).unwrap();
        self.gradient_decent.change_biases(self.biases, &summed_gradient);
        // println!("δl-1=wl^T*δl");
        self.weights.t().dot(&gradient)
    }
}