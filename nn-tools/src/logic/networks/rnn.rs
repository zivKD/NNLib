use crate::ArrView;
use ndarray::Slice;
use std::{cell::RefMut, cmp::max, time::Instant};

use ndarray::{Axis, Order, s};
use ndarray_rand::{RandomExt, rand_distr::{Normal, Uniform, uniform::UniformFloat}};
use ndarray_stats::histogram::{Grid, GridBuilder};

use crate::{Arr, DEFAULT, logic::{activations_fns::{base_activation_fn::ActivationFN,}, gradient_decents::{base_gradient_decent::GradientDecent, stochastic}, layers::{base_layer::Layer, rnn_step}, loss_fns::base_loss_fn::LossFN, utils::arr_zeros_with_shape}};

pub struct Network<'a> {
    data_set: &'a Arr,
    labels_set: &'a Arr,
    mini_batch_size: usize,
    sequence_size: usize,
    bptt_truncate: i8,
    word_dim: usize,
    hidden_dim: usize,
    gradient_decent: &'a dyn GradientDecent,
    activation_fn: &'a dyn ActivationFN,
    input_weights: &'a mut Arr,
    state_weights: &'a mut Arr,
    output_weights: &'a mut Arr,
    loss_fn: &'a dyn LossFN,
}

impl Network<'_> {
    pub fn new<'a>(
        data_set: &'a Arr,
        labels_set: &'a Arr,
        mini_batch_size: usize,
        sequence_size: usize,
        bptt_truncate: i8,
        word_dim: usize,
        hidden_dim: usize,
        gradient_decent: &'a dyn GradientDecent,
        activation_fn: &'a dyn ActivationFN,
        input_weights: &'a mut Arr,
        state_weights: &'a mut Arr,
        output_weights: &'a mut Arr,
        loss_fn: &'a dyn LossFN,
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
            output_weights,
            loss_fn,
        }
    }

    pub fn run(&mut self, print_result: bool) {
        let mut iteration = 1;
        let mut lower_bound = 0;
        let size: usize = self.sequence_size * self.word_dim;
        let mut higher_bound = size;
        while higher_bound <= (self.data_set.shape()[0] - size) {
            let mini_batch: ArrView = self.data_set.slice(s![lower_bound..higher_bound, ..]);
            let mini_batch = mini_batch
                                                            .to_shape(((size, self.mini_batch_size), Order::ColumnMajor)).unwrap().to_owned();
            let mini_batch_lbs = self.labels_set.slice_axis(
                Axis(0), 
                Slice::from((iteration-1)*self.sequence_size..iteration*self.sequence_size)
            ).to_owned();

            self.run_single_step(&mini_batch, &mini_batch_lbs, print_result);

            println!("iteration: {}", iteration);
            iteration+=1;
            lower_bound = (iteration - 1) * size;
            higher_bound = iteration * size;
        }
    }

    fn run_single_step(&mut self, inputs: &Arr, labels: &Arr, print_result: bool) {
        // let timer1 = Instant::now();
        // let prev_s_timer = Instant::now();
        let mut prev_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        // println!("init arr time: {:.2?}", prev_s_timer.elapsed());

        let mut layers = Vec::new();

        for t in 0..self.sequence_size {
            // let layer_init_timer = Instant::now();
            let mut layer = rnn_step::Init::new(
                &self.state_weights,
                &self.input_weights,
                &self.output_weights,
                self.activation_fn
            );
            // println!("layer_init time: {:.2?}", layer_init_timer.elapsed());

            // let get_input_timer = Instant::now();
            let input = self.get_input(inputs, t);
            // println!("get_input time: {:.2?}", get_input_timer.elapsed());
            // let feedforward_timer = Instant::now();
            layer.feedforward((input, prev_s.view()));
            // println!("feedforward time: {:.2?}", feedforward_timer.elapsed());
            prev_s = layer.s.clone();
            layers.push(layer);
        }

        // println!("forward time: {:.2?}", timer1.elapsed());

        let mut errors = Vec::new();
        for t in (0..self.sequence_size) {
            // let get_last_layer_timer = Instant::now();
            let last_layer_labels = self.get_last_layer_labels(&labels, t).to_owned();
            // println!("last_layer time: {:.2?}", get_last_layer_timer.elapsed());
            // let dmulv_timer = Instant::now();
            let mut dmulv = self.loss_fn.propogate(&mut DEFAULT(), &layers[t].mulv, &last_layer_labels); 
            // println!("cross_entropy_with time: {:.2?}", dmulv_timer.elapsed());
            errors.push(dmulv);
        }

        if print_result {
            let mut loss = 0.;
            for t in (0..self.sequence_size).rev() {
                loss += -errors[t].iter().filter(|f| **f < 0.).sum::<f64>() as f64;
            }

            println!("error is {}", loss / (self.mini_batch_size * self.sequence_size * self.word_dim) as f64)
        }
        
        let mut dU = arr_zeros_with_shape(self.input_weights.shape());
        let mut dW = arr_zeros_with_shape(self.state_weights.shape());
        let mut dV = arr_zeros_with_shape(self.output_weights.shape());
        let mut prev_s_t = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let diff_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);

        // let timer2 = Instant::now();
        for t in (0..self.sequence_size).rev() {
            let input = self.get_input(inputs, t);
            // let propogate_timer = Instant::now();
            let (mut dprev_s, mut dU_t, mut dW_t, dV_t) = 
                    layers[t].propogate(&input, &prev_s_t, &diff_s, &errors[t]);
            // println!("propogate time: {:.2?}", propogate_timer.elapsed());
            prev_s_t = layers[t].s.clone();
            let dmulv = arr_zeros_with_shape(&[self.word_dim, self.mini_batch_size]); 
            let bptt_amount = t as i8 - self.bptt_truncate - 1;
            let max = i8::max(0, bptt_amount) as usize;
            for i in t.checked_sub(1).unwrap_or(0)..=max {
                let input = self.get_input(inputs, i);
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

        // println!("backwards time: {:.2?}", timer2.elapsed());

        // let timer3 = Instant::now();
        self.gradient_decent.change_weights(self.input_weights, &dU);
        self.gradient_decent.change_weights(self.state_weights, &dW);
        self.gradient_decent.change_weights(self.output_weights, &dV);
        // println!("gradient time: {:.2?}", timer3.elapsed());

        // println!("total time: {:.2?}", timer1.elapsed());
    }

    fn get_input<'k>(&self, inputs: &'k Arr, t: usize) -> ArrView<'k> {
        inputs.slice_axis(Axis(0), Slice::from(t..t+self.word_dim))
    }

    fn get_last_layer_labels<'k>(&self, labels: &'k Arr, t: usize) -> ArrView<'k> {
        labels.row(t).insert_axis(Axis(0))
    }

}