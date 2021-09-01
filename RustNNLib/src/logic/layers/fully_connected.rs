use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;
use ndarray::Zip;
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
    fn feedforward(&mut self, inputs: Arr) -> Arr {
        let mut dots = inputs.dot(self.weights);
        Zip::from(&mut dots).and(&mut *self.biases).for_each(|dot, &mut bias| *dot += bias);
        self.current_weighted_inputs = dots.clone();
        let a = self.activation_fn.forward(&dots);
        self.current_activations = a.clone();
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