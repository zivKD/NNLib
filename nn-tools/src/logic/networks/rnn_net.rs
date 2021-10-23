use ndarray::Zip;
use crate::{ArrView, logic::{layers::base_layer::Layer, utils::gradient_clipping}};
use ndarray::Slice;
use std::{cell::RefMut, ops::DerefMut};
use ndarray::{Axis, Order, s};

use crate::{Arr, DEFAULT, logic::{activations_fns::{base_activation_fn::ActivationFN,}, loss_fns::base_loss_fn::LossFN, utils::arr_zeros_with_shape}};
use crate::logic::layers::rnn_step;
use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;

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
    loss_fn: &'a dyn LossFN,
}

pub struct NetworkRunParams<'a> {
    input_weights: RefMut<'a, &'a mut Arr>,
    state_weights: RefMut<'a, &'a mut Arr>,
    output_weights: RefMut<'a, &'a mut Arr>,
    state_biases: RefMut<'a, &'a mut Arr>,
    output_biases: RefMut<'a, &'a mut Arr>,
}

impl NetworkRunParams<'_> {
    pub fn new<'a>(
        input_weights: RefMut<'a, &'a mut Arr>,
        state_weights: RefMut<'a, &'a mut Arr>,
        output_weights: RefMut<'a, &'a mut Arr>,
        state_biases: RefMut<'a, &'a mut Arr>,
        output_biases: RefMut<'a, &'a mut Arr>,
    ) -> NetworkRunParams<'a> {
        NetworkRunParams {
            input_weights,
            state_weights,
            output_weights,
            state_biases,
            output_biases
        }
    }
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
            loss_fn,
        }
    }

    pub fn run(&self, smoth_loss: &mut Arr, mut params: NetworkRunParams) {
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

            self.run_single_step(&mini_batch, &mini_batch_lbs, smoth_loss, &mut params);

            iteration+=1;
            lower_bound = (iteration - 1) * size;
            higher_bound = iteration * size;
        }
    }

    pub fn run_single_step(&self, inputs: &Arr, labels: &Arr, smoth_loss: &mut Arr, params: &mut NetworkRunParams) {
        let NetworkRunParams { 
            input_weights, 
            state_weights, 
            output_weights, 
            state_biases, 
            output_biases
        } = params;

        let mut prev_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let mut layers = Vec::new();

        let mut inputs_sliced = Vec::new();
        let mut lables_sliced = Vec::new();
        for t in 0..self.sequence_size {
            inputs_sliced.push(self.get_input(inputs, t));
            lables_sliced.push(self.get_layer_lables(&labels, t).to_owned());
        }


        let mut states = Vec::new();
        let mut outputs = Vec::new();

        for t in 0..self.sequence_size {
            let layer = rnn_step::Init::new(
                &state_weights,
                &input_weights,
                &output_weights,
                &state_biases,
                &output_biases,
                self.activation_fn
            );

            let (w_frd, u_frd, add, state, output) = 
                layer.feedforward((&inputs_sliced[t], prev_s.view()));
            states.push(state);
            outputs.push(output);
            layers.push(layer);
        }


        let mut loss = arr_zeros_with_shape(smoth_loss.shape());
        for t in 0..self.sequence_size {
             loss = loss + self.loss_fn.output(&outputs[t], &lables_sliced[t]);
        }

        Zip::from(smoth_loss).and(&loss).for_each(|sl, &l| *sl = 0.999 * *sl + 0.001*l);

        let mut du = arr_zeros_with_shape(input_weights.shape());
        let mut dw = arr_zeros_with_shape(state_weights.shape());
        let mut dv = arr_zeros_with_shape(output_weights.shape());
        let mut dbs = arr_zeros_with_shape(state_biases.shape());
        let mut dbo = arr_zeros_with_shape(output_biases.shape());
        let mut prev_s_t = &arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let diff_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);

        for t in (0..self.sequence_size).rev() {
            let dmulv = self.loss_fn.propogate(&mut DEFAULT(), &outputs[t], &lables_sliced[t]); 
            let (
                mut dprev_s, 
                mut du_t, 
                mut dw_t, 
                dv_t,
                mut dbs_t,
                mut dbo_t
            ) = 
                    layers[t].propogate(&states[t], &inputs_sliced[t], &prev_s_t, &diff_s, &dmulv);
            prev_s_t = &states[t];
            let dmulv = arr_zeros_with_shape(&[self.word_dim, self.mini_batch_size]); 
            let bptt_amount = t as i8 - self.bptt_truncate - 1;
            let max = i8::max(0, bptt_amount) as usize;
            let current_index = t.checked_sub(1).unwrap_or(0);
            let ht_clone = &states[current_index];
            let prev_s_zero = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
            for i in current_index..=max {
                let input = self.get_input(inputs, i);
                let prev_s_i = if  i == 0 {
                    &prev_s_zero
                } else {
                    &ht_clone
                };

                let (
                    new_dprev_s, 
                    du_i, 
                    dw_i, 
                    _,
                    dbs_i,
                    dbo_i
                ) = 
                        layers[i].propogate(&states[i], &input, &prev_s_i, &dprev_s, &dmulv);

                dprev_s = new_dprev_s;
                du_t = du_t + du_i;
                dw_t = dw_t + dw_i;
                dbs_t = dbs_t + dbs_i;
                dbo_t = dbo_t + dbo_i;
            }

            du = du + du_t;
            dw = dw + dw_t;
            dv = dv + dv_t;
            dbs = dbs + dbs_t;
            dbo = dbo + dbo_t;
        }

        let min_clipping_value = -5.;
        let max_clipping_value = 5.;
        let clipped_du = gradient_clipping(&du, min_clipping_value, max_clipping_value);
        let clipped_dw = gradient_clipping(&dw, min_clipping_value, max_clipping_value);
        let clipped_dv = gradient_clipping(&dv, min_clipping_value, max_clipping_value);
        let cilpped_dbs = gradient_clipping(&dbs, min_clipping_value, max_clipping_value);
        let cilpped_dbo = gradient_clipping(&dbo, min_clipping_value, max_clipping_value);

        self.gradient_decent.change_weights(input_weights.deref_mut(), &clipped_du);
        self.gradient_decent.change_weights(state_weights.deref_mut(), &clipped_dw);
        self.gradient_decent.change_weights(output_weights.deref_mut(), &clipped_dv);
        self.gradient_decent.change_biases(state_biases.deref_mut(), &cilpped_dbs);
        self.gradient_decent.change_weights(output_biases.deref_mut(), &cilpped_dbo);
    }

    fn get_input<'k>(&self, inputs: &'k Arr, t: usize) -> ArrView<'k> {
        inputs.slice_axis(Axis(0), Slice::from((t*self.word_dim)..(t+1)*self.word_dim))
    }

    fn get_layer_lables<'k>(&self, labels: &'k Arr, t: usize) -> ArrView<'k> {
        labels.row(t).insert_axis(Axis(0))
    }

}