use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::utils::{repeated_axis_zero};
use crate::{Arr, ArrView};
use ndarray::{Axis, Zip};
use ndarray_stats::QuantileExt;
use crate::logic::layers::base_layer::Layer;
use crate::logic::activations_fns::{self, softmax};

pub struct Init<'a> {
    state_weights: &'a mut Arr,
    input_weights: &'a mut Arr,
    output_weights: &'a mut Arr,
    biases: &'a mut Arr,
    hidden_activation_fn: &'a dyn ActivationFN,
    output_activation_fn: &'a dyn ActivationFN,
    gradient_decent: &'a dyn GradientDecent,
}

impl Init<'_> {
    pub fn new<'a>(
        state_weights: &'a mut Arr,
        input_weights: &'a mut Arr,
        output_weights: &'a mut Arr,
        biases: &'a mut Arr, 
        hidden_activation_fn: &'a dyn ActivationFN,
        output_activation_fn: &'a dyn ActivationFN,
        gradient_decent: &'a dyn GradientDecent
    ) -> Init<'a> {
        Init {
            state_weights,
            input_weights,
            output_weights,
            biases,
            hidden_activation_fn,
            output_activation_fn,
            gradient_decent,
        }
    }
}

impl Init<'_> {
    fn feedforward(&mut self, inputs: (ArrView, ArrView)) -> (Arr, Arr) {
        let (inputs_embadding, hidden_state) = inputs;
        let w_frd = self.state_weights.dot(&hidden_state);
        let u_frd = self.input_weights.dot(&inputs_embadding);
        let sum_s = w_frd + u_frd;
        let ht_activated = self.hidden_activation_fn.forward(&sum_s);
        let yt_unactivated = self.output_weights.dot(&ht_activated);
        let yt_activated = self.output_activation_fn.forward(&yt_unactivated);
        (ht_activated, yt_activated)
    }

    fn propogate(&mut self, gradient: Arr, activations: ArrView) -> Arr {
        Arr::default((1,1))
    }

    fn tanh(&self, arr: Arr) -> Arr {
        arr.map(|x| {
            x.tanh()
        })
    }

}

