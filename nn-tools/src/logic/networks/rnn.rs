use std::{cell::RefMut, cmp::max};

use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform}};
use ndarray_stats::histogram::{Grid, GridBuilder};

use crate::{Arr, logic::{activations_fns::{base_activation_fn::ActivationFN, tanh}, gradient_decents::{base_gradient_decent::GradientDecent, stochastic}, layers::{base_layer::Layer, rnn_step}, loss_fns::base_loss_fn::LossFN, utils::arr_zeros_with_shape}};

pub struct Network<'a> {
    data_set : &'a Arr,
    lbl_set: &'a Arr,
    mini_batch_size: usize,
    sequence_size: usize,
    bptt_truncate: usize,
    word_dim: usize,
    gradient_decent: &'a dyn GradientDecent,
    loss_fn: &'a dyn LossFN,
    activation_fn: &'a dyn ActivationFN
}

impl Network<'_> {
    pub fn new<'a>(
        data_set : &'a Arr,
        lbl_set: &'a Arr,
        mini_batch_size: usize,
        sequence_size: usize,
        bptt_truncate: usize,
        word_dim: usize,
        gradient_decent: &'a dyn GradientDecent,
        loss_fn: &'a dyn LossFN,
        activation_fn: &'a dyn ActivationFN,
    ) -> Network<'a> {
        Network {
            data_set,
            lbl_set,
            mini_batch_size,
            sequence_size,
            bptt_truncate,
            word_dim,
            gradient_decent,
            loss_fn,
            activation_fn
        }
    }

    pub fn run(&mut self, inputs: &Arr, print_result: bool) {
        let mut inputs_weights: Arr = Arr::random((1,1), Normal::new(0., 0.03).unwrap());
        let mut state_weights = Arr::random((1,1), Normal::new(0., 0.03).unwrap());
        let mut output_weights = Arr::random((1,1), Normal::new(0., 0.03).unwrap());
        let tanh = tanh::Init {};
        let mut prev_s = Arr::default((1,1));
        let mut layers = Vec::new();
        for t in (0..self.sequence_size) {
            let mut layer = rnn_step::Init::new(
                &state_weights,
                &inputs_weights,
                &output_weights,
                &tanh,
            );

            let mut input = Arr::zeros((self.sequence_size, self.mini_batch_size));
            input[(inputs[(t,0)] as usize, 1)] = 1.;
            layer.feedforward((input.view(), prev_s.view()));
            prev_s = layer.s.clone();
            layers.push(layer);
        }

        let mut dU = arr_zeros_with_shape(inputs_weights.shape());
        let mut dW = arr_zeros_with_shape(state_weights.shape());
        let mut dV = arr_zeros_with_shape(output_weights.shape());
        let mut prev_s_t = Arr::zeros((1,1));
        let diff_s = Arr::zeros((1,1));
        let default = Arr::zeros((1,1));

        for t in self.sequence_size..0 {
            let mut dmulv = Arr::zeros((1,1)); // Actually loss fn derviative
            let mut input = Arr::zeros((self.sequence_size, self.mini_batch_size));
            input[(inputs[(t,0)] as usize, 1)] = 1.;
            let (dprev_s, mut dU_t, mut dW_t, dV_t) = 
                    layers[t].propogate(&input, &prev_s_t, &diff_s, &dmulv);
            prev_s_t = dprev_s;
            dmulv = Arr::zeros((1,1)); // Actually loss fn derviative
            let max = i32::max(0, (t - self.bptt_truncate - 1) as i32) as usize;
            for i in t-1..max {
                let mut input = Arr::zeros((self.sequence_size, self.mini_batch_size));
                input[(inputs[(t,0)] as usize, 1)] = 1.;
                let dprev_s_i = if  i == 0 {
                    default.clone()
                } else {
                    layers[i-1].s.clone()
                };

                let (dprev_s, dU_i, dW_i, dV_i) = 
                        layers[i].propogate(&input, &dprev_s_i, &prev_s_t, &dmulv);
                dU_t = dU_t + dU_i;
                dW_t = dW_t + dW_i;
            }

            dU = dU + dU_t;
            dW = dW + dW_t;
            dV = dV + dV_t;
        }

        self.gradient_decent.change_weights(&mut inputs_weights, &dU);
        self.gradient_decent.change_weights(&mut state_weights, &dW);
        self.gradient_decent.change_weights(&mut output_weights, &dV);
    }
}