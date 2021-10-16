use std::time::Instant;
use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::utils::{repeated_axis_zero};
use crate::{Arr, ArrView, DEFAULT};
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
    pub fn feedforward(&mut self, inputs: (ArrView, ArrView)) {
        // let decontruction_timer = Instant::now();
        let (inputs_embadding, hidden_state) = inputs;
        // println!("deconstruction time: {:.2?}", decontruction_timer.elapsed());
        // let w_frd_timer = Instant::now();
        let w_frd = self.state_weights.dot(&hidden_state);
        // println!("w_frd time: {:.2?}", w_frd_timer.elapsed());
        // let u_frd_timer = Instant::now();
        let u_frd = self.input_weights.dot(&inputs_embadding);
        // println!("u_frd time: {:.2?}", u_frd_timer.elapsed());
        // let sum_s_timer = Instant::now();
        let sum_s = &w_frd + &u_frd;
        // println!("sum_s time: {:.2?}", sum_s_timer.elapsed());
        // let  hidden_activation_timer = Instant::now();
        let ht_activated = self.hidden_activation_fn.forward(&sum_s);
        // println!("hidden activation time: {:.2?}", hidden_activation_timer.elapsed());
        // let yt_timer = Instant::now();
        let yt= self.output_weights.dot(&ht_activated);
        // println!("yt time: {:.2?}", yt_timer.elapsed());
        // let assign_timer = Instant::now();
        self.mulw = w_frd;
        self.mulu = u_frd;
        self.add = sum_s;
        self.s = ht_activated;
        self.mulv = yt;
        // println!("assign time: {:.2?}", assign_timer.elapsed());
    }

    pub fn propogate(
        &mut self, inputs: &ArrView, prev_s: &Arr, diff_s: &Arr, dmulv: &Arr) -> 
        (Arr, Arr, Arr, Arr) {
        // let dv_timer = Instant::now();
        let (dV, dsv) = self.multiplication_backward(self.output_weights, &self.s, dmulv);
        // println!("dv time: {:.2?}", dv_timer.elapsed());
        // let ds_timer = Instant::now();
        let ds = dsv + diff_s;
        // println!("ds time: {:.2?}", ds_timer.elapsed());
        // let dadd_timer = Instant::now();
        let dadd = self.hidden_activation_fn.propogate(&self.add) * ds;
        // println!("dadd time: {:.2?}", dadd_timer.elapsed());
        // let dmulw_timer = Instant::now();
        // let (dmulw, dmulu) = self.add_backward(&self.mulu, &self.mulw, &dadd);
        // println!("dmulw time: {:.2?}", dmulw_timer.elapsed());
        // let dw_timer = Instant::now();
        let (dW, dprev_s) = self.multiplication_backward(self.state_weights, prev_s, &dadd);
        // println!("dw time: {:.2?}", dw_timer.elapsed());
        // let du_timer = Instant::now();
        let (dU, dx) = self.multiplication_backward(self.input_weights, &inputs.to_owned(), &dadd);
        // println!("du time: {:.2?}", du_timer.elapsed());

        (dprev_s, dU, dW, dV)
    }

    fn multiplication_backward(&self, weights: &Arr, x: &Arr, dz: &Arr) -> (Arr, Arr) {
        (dz.dot(&x.t()), weights.t().dot(dz))
    }
}
