use std::time::Instant;
use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::utils::{repeated_axis_zero};
use crate::{Arr, ArrView, DEFAULT};
use ndarray::{Axis, Zip};
use ndarray_stats::QuantileExt;
use crate::logic::layers::base_layer::Layer;
use crate::logic::activations_fns::{self};

pub struct Init<'a> {
    state_weights: &'a Arr,
    input_weights: &'a Arr,
    output_weights: &'a Arr,
    hidden_activation_fn: &'a dyn ActivationFN,
    mulu : Arr,
    mulw : Arr,
    add : Arr,
    pub s: Arr,
    pub mulv: Arr,
}

impl Init<'_> {
    pub fn new<'a>(
        state_weights: &'a Arr,
        input_weights: &'a Arr,
        output_weights: &'a Arr,
        hidden_activation_fn: &'a dyn ActivationFN,
    ) -> Init<'a> {
        let default_val = DEFAULT();
        Init {
            state_weights,
            input_weights,
            output_weights,
            hidden_activation_fn,
            mulu: default_val.clone(),
            mulw: default_val.clone(),
            add: default_val.clone(),
            s: default_val.clone(),
            mulv: default_val.clone()
        }
    }
}

impl Init<'_> {
    pub fn feedforward(&mut self, inputs: (&ArrView, ArrView)) {
        let (inputs_embadding, hidden_state) = inputs;
        let w_frd = self.state_weights.dot(&hidden_state);
        let u_frd = self.input_weights.dot(inputs_embadding);
        let sum_s = &w_frd + &u_frd;
        let ht_activated = self.hidden_activation_fn.forward(&sum_s);
        let yt= self.output_weights.dot(&ht_activated);
        self.mulw = w_frd;
        self.mulu = u_frd;
        self.add = sum_s;
        self.s = ht_activated;
        self.mulv = yt;
    }

    pub fn propogate(
        &mut self, inputs: &ArrView, prev_s: &Arr, diff_s: &Arr, dmulv: &Arr) -> 
        (Arr, Arr, Arr, Arr) {
        let (dV, dsv) = 
            self.multiplication_backward(self.output_weights, &self.s, dmulv);
        let ds = dsv + diff_s;
        let dadd = self.hidden_activation_fn.propogate(&self.add) * ds;
        let (dW, dprev_s) = self.multiplication_backward(self.state_weights, prev_s, &dadd);
        let (dU, dx) = self.multiplication_backward(self.input_weights, &inputs.to_owned(), &dadd);

        (dprev_s, dU, dW, dV)
    }

    fn multiplication_backward(&self, weights: &Arr, x: &Arr, dz: &Arr) -> (Arr, Arr) {
        (dz.dot(&x.t()), weights.t().dot(dz))
    }
}
