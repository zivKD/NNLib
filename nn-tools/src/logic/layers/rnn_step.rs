use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::utils::{repeated_axis_zero};
use crate::{Arr, ArrView};
use ndarray::{Axis, Zip};
use ndarray_stats::QuantileExt;
use crate::logic::layers::base_layer::Layer;
use crate::logic::activations_fns::{self, softmax};

pub struct Init<'a> {
    state_weights: &'a Arr,
    input_weights: &'a Arr,
    output_weights: &'a Arr,
    hidden_activation_fn: &'a dyn ActivationFN,
    mulu : Arr,
    mulw : Arr,
    add : Arr,
    pub s: Arr,
    mulv: Arr,
}

impl Init<'_> {
    pub fn new<'a>(
        state_weights: &'a Arr,
        input_weights: &'a Arr,
        output_weights: &'a Arr,
        hidden_activation_fn: &'a dyn ActivationFN,
    ) -> Init<'a> {
        let default = Arr::default((1,1));
        Init {
            state_weights,
            input_weights,
            output_weights,
            hidden_activation_fn,
            mulu: Arr::default((1,1)),
            mulw: Arr::default((1,1)),
            add: Arr::default((1,1)),
            s: Arr::default((1,1)),
            mulv: Arr::default((1,1))
        }
    }
}

impl Init<'_> {
    pub fn feedforward(&mut self, inputs: (ArrView, ArrView)) {
        let (inputs_embadding, hidden_state) = inputs;
        let w_frd = self.state_weights.dot(&hidden_state);
        let u_frd = self.input_weights.dot(&inputs_embadding);
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
        &mut self, inputs: &Arr, prev_s: &Arr, diff_s: &Arr, dmulv: &Arr) -> 
        (Arr, Arr, Arr, Arr) {
        let (dV, dsv) = self.multiplication_backward(self.output_weights, &self.s, dmulv);
        let ds = dsv + diff_s;
        let dadd = self.hidden_activation_fn.propogate(&self.add) * ds;
        let (dmulw, dmulu) = self.add_backward(&self.mulu, &self.mulw, &dadd);
        let (dW, dprev_s) = self.multiplication_backward(self.state_weights, prev_s, &dmulw);
        let (dU, dx) = self.multiplication_backward(self.input_weights, &inputs.to_owned(), &dmulu);

        (dprev_s, dU, dW, dV)
    }

    fn multiplication_backward(&self, weights: &Arr, x: &Arr, dz: &Arr) -> (Arr, Arr) {
        (dz.dot(&x.t()), weights.t().dot(dz))
    }

    fn add_backward(&self, x1: &Arr, x2: &Arr, dz: &Arr) -> (Arr, Arr) {
        let x1_shape = x1.shape();
        let x2_shape = x2.shape();
        (dz * Arr::ones((x1_shape[0], x1_shape[1])), dz * Arr::ones((x2_shape[0], x2_shape[1]))) 
    }
}
