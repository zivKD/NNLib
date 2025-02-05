use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::utils::{repeated_axis_zero};
use crate::{Arr, ArrView};
use ndarray::{Axis, Zip};
use crate::logic::layers::base_layer::Layer;

pub struct Init<'a> {
    weights: &'a mut Arr,
    biases: &'a mut Arr,
    gradient_decent: &'a dyn GradientDecent,
}

impl Init<'_> {
    pub fn new<'a>(
        weights: &'a mut Arr, 
        biases: &'a mut Arr, 
        gradient_decent: &'a dyn GradientDecent
    ) -> Init<'a> {
        Init {
            weights,
            biases,
            gradient_decent,
        }
    }
}

impl Layer for Init<'_> {
    fn feedforward(&mut self, inputs: ArrView) -> Arr {
        let mut dots = self.weights.dot(&inputs);
        let mut repeated_biases = repeated_axis_zero(self.biases,&(dots.shape()[0], dots.shape()[1]));
        Zip::from(&mut dots).and(&mut repeated_biases).for_each(|dot, &mut bias| *dot += bias);
        dots
    }

    fn propogate(&mut self, gradient: Arr, activations: ArrView) -> Arr {
        // NOT THE STANDARD VERSION δl-1=wl^T*δl
        let next_gradiet =  self.weights.t().dot(&gradient);
        // ∂C∂wl=δl*al-1^T Reason for transpose is matrix form, originally it's δl_j*al-1_k (so jth neuron in l layer to kth neuron in l layer)
        self.gradient_decent.change_weights(self.weights, &gradient.dot(&activations.t()));
        // CURENTLY SUM, MABYE BETTER IDEA
        let summed_gradient = gradient.sum_axis(Axis(1)).into_shape((gradient.shape()[0], 1)).unwrap();
        // ∂C∂bl=δl
        self.gradient_decent.change_biases(self.biases, &summed_gradient);
        next_gradiet
    }
}