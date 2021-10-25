use std::cell::{RefCell, RefMut};

use ndarray::Zip;
use crate::{ArrView};
use ndarray::Slice;
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
    activation_fn: &'a dyn ActivationFN,
    loss_fn: &'a dyn LossFN,
}

pub struct NetworkRunParams<'a> {
    input_weights: &'a RefCell<&'a mut Arr>,
    state_weights: &'a RefCell<&'a mut Arr>,
    output_weights: &'a RefCell<&'a mut Arr>,
    state_biases: &'a RefCell<&'a mut Arr>,
    output_biases: &'a RefCell<&'a mut Arr>,
}

impl NetworkRunParams<'_> {
    pub fn new<'a>(
        input_weights: &'a RefCell<&'a mut Arr>,
        state_weights: &'a RefCell<&'a mut Arr>,
        output_weights: &'a RefCell<&'a mut Arr>,
        state_biases: &'a RefCell<&'a mut Arr>,
        output_biases: &'a RefCell<&'a mut Arr>,
    ) -> NetworkRunParams<'a> {
        NetworkRunParams {
            input_weights,
            state_weights,
            output_weights,
            state_biases,
            output_biases,
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
            activation_fn,
            loss_fn,
        }
    }

    pub fn run<T: FnMut((Arr,Arr,Arr,Arr,Arr))>(&self, smoth_loss: &mut Arr, mut gradient_decent: T, mut params: NetworkRunParams) {
        let NetworkRunParams { 
            input_weights,
            state_weights, 
            output_weights, 
            state_biases, 
            output_biases
        } = params;


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

            let def_value = DEFAULT();
            let mut gradients = (def_value.clone(), def_value.clone(), def_value.clone(), def_value.clone(), def_value.clone());
            {

                let iw = input_weights.borrow();
                let sw = state_weights.borrow();
                let ow = output_weights.borrow();
                let sb = state_biases.borrow(); 
                let ob = output_biases.borrow();

                let shapes = (
                    iw.shape(), 
                    sw.shape(), 
                    ow.shape(), 
                    sb.shape(), 
                    ob.shape()
                );

                let rnn_unit = rnn_step::Init::new(
                    &sw, 
                    &iw, 
                    &ow, 
                    &sb,
                    &ob,
                    self.activation_fn
                );
                let mut inputs_sliced = Vec::new();
                let mut labels_sliced = Vec::new();
                for t in 0..self.sequence_size {
                    inputs_sliced.push(self.get_input(&mini_batch, t));
                    labels_sliced.push(self.get_layer_lables(&mini_batch_lbs, t).to_owned());
                }

                let (states, outputs) = self.forward(&inputs_sliced, &rnn_unit);
                self.calculate_loss(smoth_loss, &labels_sliced, &outputs);
                gradients = self.propogate(&inputs_sliced, &rnn_unit, &labels_sliced, &outputs, &states, shapes);
            }

            gradient_decent(gradients);

            iteration+=1;
            lower_bound = (iteration - 1) * size;
            higher_bound = iteration * size;
        }
    }


    fn forward(&self, inputs: &Vec<ArrView>, rnn_unit: &rnn_step::Init) -> (Vec<Arr>, Vec<Arr>) {
        let mut prev_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let mut states = Vec::new();
        let mut outputs = Vec::new();

        for t in 0..self.sequence_size {
            let (w_frd, u_frd, add, state, output) = 
                rnn_unit.feedforward((&inputs[t], prev_s.view()));
            states.push(state);
            outputs.push(output);
        }

        (states, outputs)
    }

    fn calculate_loss(&self, smoth_loss: &mut Arr, labels: &Vec<Arr>, outputs: &Vec<Arr>) {
        let mut loss = arr_zeros_with_shape(smoth_loss.shape());
        for t in 0..self.sequence_size {
             loss = loss + self.loss_fn.output(&outputs[t], &labels[t]);
        }

        Zip::from(smoth_loss).and(&loss).for_each(|sl, &l| *sl = 0.999 * *sl + 0.001*l);
    }

    fn propogate(&self, inputs: &Vec<ArrView>, rnn_unit: &rnn_step::Init, labels: &Vec<Arr>, outputs: &Vec<Arr>, states: &Vec<Arr>, shapes: (&[usize],&[usize],&[usize],&[usize],&[usize],)) -> 
        (Arr, Arr, Arr, Arr, Arr) {
        
        let (iw_shape, sw_shape, ow_shape, sb_shape, ob_shape) = shapes;
        let mut du = arr_zeros_with_shape(iw_shape);
        let mut dw = arr_zeros_with_shape(sw_shape);
        let mut dv = arr_zeros_with_shape(ow_shape);
        let mut dbs = arr_zeros_with_shape(sb_shape);
        let mut dbo = arr_zeros_with_shape(ob_shape);
        let mut prev_s_t = &arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
        let diff_s = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);

        for t in (0..self.sequence_size).rev() {
            let dmulv = self.loss_fn.propogate(&mut DEFAULT(), &outputs[t], &labels[t]); 
            let (
                mut dprev_s, 
                mut du_t, 
                mut dw_t, 
                dv_t,
                mut dbs_t,
                mut dbo_t
            ) = 
                    rnn_unit.propogate(&states[t], &inputs[t], &prev_s_t, &diff_s, &dmulv);
            prev_s_t = &states[t];
            let dmulv = arr_zeros_with_shape(&[self.word_dim, self.mini_batch_size]); 
            let bptt_amount = t as i8 - self.bptt_truncate - 1;
            let max = i8::max(0, bptt_amount) as usize;
            let current_index = t.checked_sub(1).unwrap_or(0);
            let ht_clone = &states[current_index];
            let prev_s_zero = arr_zeros_with_shape(&[self.hidden_dim, self.mini_batch_size]);
            for i in current_index..=max {
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
                        rnn_unit.propogate(&states[i], &inputs[i], &prev_s_i, &dprev_s, &dmulv);

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

        (du, dw, dv, dbs, dbo)
    }

    fn get_input<'k>(&self, inputs: &'k Arr, t: usize) -> ArrView<'k> {
        inputs.slice_axis(Axis(0), Slice::from((t*self.word_dim)..(t+1)*self.word_dim))
    }

    fn get_layer_lables<'k>(&self, labels: &'k Arr, t: usize) -> ArrView<'k> {
        labels.row(t).insert_axis(Axis(0))
    }

}