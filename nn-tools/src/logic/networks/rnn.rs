use ndarray::Slice;
use crate::ArrView;
use std::{cell::RefMut, cmp::max};

use ndarray::{Axis, Order, s};
use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform, uniform::UniformFloat}};
use ndarray_stats::histogram::{Grid, GridBuilder};

use crate::{Arr, DEFAULT, logic::{activations_fns::{base_activation_fn::ActivationFN, softmax, tanh}, gradient_decents::{base_gradient_decent::GradientDecent, stochastic}, layers::{base_layer::Layer, rnn_step}, loss_fns::base_loss_fn::LossFN, utils::arr_zeros_with_shape}};

pub struct Network<'a> {
    data_set: &'a Arr,
    labels_set: &'a Arr,
    mini_batch_size: usize,
    sequence_size: usize,
    bptt_truncate: usize,
    word_dim: usize,
    hidden_dim: usize,
    gradient_decent: &'a dyn GradientDecent,
    activation_fn: &'a dyn ActivationFN,
    input_weights: &'a mut Arr,
    state_weights: &'a mut Arr,
    output_weights: &'a mut Arr,
}

impl Network<'_> {
    pub fn new<'a>(
        data_set: &'a Arr,
        labels_set: &'a Arr,
        mini_batch_size: usize,
        sequence_size: usize,
        bptt_truncate: usize,
        word_dim: usize,
        hidden_dim: usize,
        gradient_decent: &'a dyn GradientDecent,
        activation_fn: &'a dyn ActivationFN,
        input_weights: &'a mut Arr,
        state_weights: &'a mut Arr,
        output_weights: &'a mut Arr,
    ) -> Network<'a> {
        Network {
            data_set,
            labels_set,
            mini_batch_size,
            sequence_size,
            bptt_truncate,
            word_dim,
            hidden_dim,
            gradient_decent,
            activation_fn,
            input_weights,
            state_weights,
            output_weights
        }
    }

    pub fn run(&mut self, print_result: bool) {
        let mut iteration = 1;
        let mut lower_bound = 0;
        let mut higher_bound = self.mini_batch_size * self.sequence_size;
        while higher_bound <= self.data_set.len() * self.sequence_size {
            let mini_batch: ArrView = self.data_set.slice(s![lower_bound..higher_bound, ..]);
            let mini_batch = mini_batch
                                                            .to_shape(((self.sequence_size, self.mini_batch_size), Order::ColumnMajor)).unwrap().to_owned();
            let mini_batch_lbs = self.labels_set.slice_axis(
                Axis(1), 
                Slice::from((iteration-1)*self.mini_batch_size..iteration*self.mini_batch_size)
            ).to_owned();
            println!("mini batch shape {:?}", mini_batch.shape());
            println!("mini batch lbls shape {:?}", mini_batch_lbs.shape());

            // self.run_single_step(&mini_batch, &mini_batch_lbs, print_result);

            iteration+=1;
            lower_bound = (iteration - 1) * (self.mini_batch_size * self.sequence_size);
            higher_bound = iteration * self.mini_batch_size * self.sequence_size;
        }
    }

    fn run_single_step(&mut self, inputs: &Arr, lablels: &Arr, print_result: bool) {
        let mut prev_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let mut layers = Vec::new();

        for t in 0..self.sequence_size {
            let mut layer = rnn_step::Init::new(
                &self.state_weights,
                &self.input_weights,
                &self.output_weights,
                self.activation_fn
            );

            let input = self.get_input(inputs, t);
            layer.feedforward((input.view(), prev_s.view()));
            prev_s = layer.s.clone();
            layers.push(layer);
        }

        let mut dU = arr_zeros_with_shape(self.input_weights.shape());
        let mut dW = arr_zeros_with_shape(self.state_weights.shape());
        let mut dV = arr_zeros_with_shape(self.output_weights.shape());
        let mut prev_s_t = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let diff_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);

        for t in self.sequence_size..0 {
            let mut dmulv = self.cross_entropy_with_softmax_propgate(&layers[t].mulv, lablels);
            let input = self.get_input(inputs, t);
            let (mut dprev_s, mut dU_t, mut dW_t, dV_t) = 
                    layers[t].propogate(&input, &prev_s_t, &diff_s, &dmulv);
            prev_s_t = layers[t].s.clone();
            dmulv = arr_zeros_with_shape(&[self.word_dim, self.mini_batch_size]); 
            let max = i32::max(0, (t - self.bptt_truncate - 1) as i32) as usize;
            for i in t-1..max {
                let input = self.get_input(inputs, t);
                let prev_s_i = if  i == 0 {
                    arr_zeros_with_shape(&[self.hidden_dim, 1])
                } else {
                    layers[i-1].s.clone()
                };

                let (new_dprev_s, dU_i, dW_i, dV_i) = 
                        layers[i].propogate(&input, &dprev_s, &prev_s_i, &dmulv);

                dprev_s = new_dprev_s;
                dU_t = dU_t + dU_i;
                dW_t = dW_t + dW_i;
            }

            dU = dU + dU_t;
            dW = dW + dW_t;
            dV = dV + dV_t;
        }

        self.gradient_decent.change_weights(self.input_weights, &dU);
        self.gradient_decent.change_weights(self.state_weights, &dW);
        self.gradient_decent.change_weights(self.output_weights, &dV);
    }

    fn get_input(&self, inputs: &Arr, t: usize) -> Arr {
        let mut input = arr_zeros_with_shape(&[self.word_dim, self.mini_batch_size]);
        (0..self.mini_batch_size).for_each(|i| input[(inputs[(t,i)] as usize, i)] = 1.);
        input
    }

    fn cross_entropy_with_softmax_propgate(&self, a: &Arr, y: &Arr) -> Arr {
        let softmax = softmax::Init {};
        let mut probs = softmax.forward(a);
        y.iter().enumerate().for_each(|(index, value)| {
            probs[(index, *value as usize)] -= 1.;
        });
        probs
    }
}