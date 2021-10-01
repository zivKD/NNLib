use ndarray::Order;
use ndarray::Slice;
use ndarray::Axis;
use core::cell::RefMut;
use ndarray::{Zip, s};
use ndarray_stats::QuantileExt;

use crate::{Arr, ArrView};

use super::{layers::base_layer::Layer, loss_fns::base_loss_fn::LossFN};

pub struct Network<'a> {
    data_set : &'a Arr,
    lbl_set: &'a Arr,
    mini_batch_size: usize,
    inputs_size: usize,
    data_set_size: usize,
    layers: RefMut<'a, Vec<&'a mut dyn Layer>>,
    loss_fn: &'a dyn LossFN,
}

impl Network<'_> {
    pub fn new<'a>(
        data_set : &'a Arr,
        lbl_set: &'a Arr,
        mini_batch_size: usize,
        inputs_size: usize,
        data_set_size: usize,
        layers: RefMut<'a, Vec<&'a mut dyn Layer>>,
        loss_fn: &'a dyn LossFN,
    ) -> Network<'a> {
        Network {
            data_set,
            lbl_set,
            mini_batch_size,
            inputs_size,
            data_set_size,
            layers,
            loss_fn
        }
    }

    // mnb 0 at 0 at 523: 0.80859 + mnb 1 at 1 at 519: 0.50781

    pub fn run(&mut self, print_result: bool) {
        let mut iteration = 1;
        let mut lower_bound = 0;
        let mut higher_bound = self.mini_batch_size * self.inputs_size;
        while higher_bound <= self.data_set_size*self.inputs_size {
            let mini_batch: ArrView = self.data_set.slice(s![lower_bound..higher_bound, ..]);
            let mini_batch = mini_batch
                                                            .to_shape(((self.inputs_size, self.mini_batch_size), Order::ColumnMajor)).unwrap();
            let mini_batch_lbs = self.lbl_set.slice_axis(
                Axis(1), 
                Slice::from((iteration-1)*self.mini_batch_size..iteration*self.mini_batch_size)
            ).to_owned();
            let mut activations = vec!();
            let inputs = mini_batch.to_owned();
            activations.push(inputs);
            let mut i = 0;
            for layer in self.layers.iter_mut() {
                let new_inputs = layer.feedforward(activations[i].view());
                activations.push(new_inputs);
                i+=1;
            }

            let mut i = activations.len() - 1;

            if print_result {
                let accuracy= Zip::from(activations[i].columns()).and(mini_batch_lbs.columns()).
                    map_collect(|x, y| {
                        let max_index_in_outputs = x.argmax().unwrap();
                        let max_index_in_lbls = y.argmax().unwrap();
                        max_index_in_outputs == max_index_in_lbls
                    }).
                    iter().filter(|x| **x).collect::<Vec<&bool>>().len();

                let success_percentage = (accuracy as f64 /self.mini_batch_size as f64) * 100.;
                println!("network accuracy: {}%", success_percentage);
            } else {
                let mut error = self.loss_fn.propogate(&mut activations[i], &mini_batch_lbs);

                for layer in self.layers.iter_mut().rev() {
                    i-=1;
                    error = layer.propogate(error, activations[i].view());
                }
            }

            iteration+=1;
            lower_bound = (iteration - 1) * (self.mini_batch_size * self.inputs_size);
            higher_bound = iteration * self.mini_batch_size * self.inputs_size;
        }
    }
}