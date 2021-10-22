use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::{Arr, ArrView, DEFAULT};

pub struct Init<'a> {
    state_weights: &'a Arr,
    input_weights: &'a Arr,
    output_weights: &'a Arr,
    state_biases: &'a Arr,
    output_biases: &'a Arr,
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
        state_biases: &'a Arr,
        output_biases: &'a Arr,
        hidden_activation_fn: &'a dyn ActivationFN,
    ) -> Init<'a> {
        let default_val = DEFAULT();
        Init {
            state_weights,
            input_weights,
            output_weights,
            state_biases,
            output_biases,
            hidden_activation_fn,
            mulu: default_val.clone(),
            mulw: default_val.clone(),
            add: default_val.clone(),
            s: default_val.clone(),
            mulv: default_val.clone(),
        }
    }
}

impl Init<'_> {
    pub fn feedforward(&mut self, inputs: (&ArrView, ArrView)) {
        let (inputs_embadding, hidden_state) = inputs;
        let u_frd = self.input_weights.dot(inputs_embadding);
        let w_frd = self.state_weights.dot(&hidden_state);
        let sum_s = &w_frd + &u_frd + self.state_biases;
        let ht_activated = self.hidden_activation_fn.forward(&sum_s);
        let yt= self.output_weights.dot(&ht_activated) + self.output_biases;
        self.mulw = w_frd;
        self.mulu = u_frd;
        self.add = sum_s;
        self.s = ht_activated;
        self.mulv = yt;
    }

    pub fn propogate(
        &mut self, inputs: &ArrView, prev_s: &Arr, diff_s: &Arr, dmulv: &Arr) -> 
        (Arr, Arr, Arr, Arr, Arr, Arr) {
        let (dv, dsv) = 
            self.multiplication_backward(self.output_weights, &self.s, dmulv);
        let ds = dsv + diff_s;
        let dadd = self.hidden_activation_fn.propogate(&self.add) * ds;
        let (dw, dprev_s) = self.multiplication_backward(self.state_weights, prev_s, &dadd);
        let (du, _) = self.multiplication_backward(self.input_weights, &inputs.to_owned(), &dadd);

        (dprev_s, du, dw, dv,dadd,dmulv.clone())
    }

    fn multiplication_backward(&self, weights: &Arr, x: &Arr, dz: &Arr) -> (Arr, Arr) {
        (dz.dot(&x.t()), weights.t().dot(dz))
    }
}
