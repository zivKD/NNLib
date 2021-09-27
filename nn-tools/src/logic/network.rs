use ndarray::s;

use crate::{Arr, ArrView};

use super::{layers::base_layer::Layer, loss_fns::base_loss_fn::LossFN};

pub struct Network<'a> {
    data_set : Arr,
    lbl_set: Arr,
    mini_batch_size: usize,
    inputs_size: usize,
    data_set_size: usize,
    layers: Vec<&'a mut Layer>,
    loss_fn: &'a LossFN,
}

impl Network<'_> {
    pub fn new<'a>(
        data_set : Arr,
        lbl_set: Arr,
        mini_batch_size: usize,
        inputs_size: usize,
        data_set_size: usize,
        layers: Vec<&'a mut Layer>,
        loss_fn: &'a LossFN
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

    pub fn run(&mut self) {
        let mut iteration = 1;
        let mut lower_bound = 0;
        let mut higher_bound = self.mini_batch_size * self.inputs_size;
        while higher_bound < self.data_set_size*self.inputs_size {
            println!("running: {}", iteration);
            let mini_batch: ArrView = self.data_set.slice(s![lower_bound..higher_bound, ..]);
            let mini_batch = mini_batch
                                                            .into_shape((self.inputs_size, self.mini_batch_size)).unwrap();
            let mini_batch_lbs = self.lbl_set.slice(s![(iteration-1)*self.mini_batch_size..iteration*self.mini_batch_size, ..]).to_owned();
            let mut activations = vec!();
            let inputs = mini_batch.to_owned();
            activations.push(inputs);
            let mut i = 0;
            for layer in &mut self.layers {
                let new_inputs = layer.feedforward(activations[i].view());
                activations.push(new_inputs);
                i+=1;
            }

            let mut i = activations.len() - 1;
            let mut error = self.loss_fn.propogate(&mut activations[i], &mini_batch_lbs);

            for layer in &mut self.layers.iter_mut().rev() {
                i-=1;
                error = layer.propogate(error, activations[i].view());
            }

            iteration+=1;
            lower_bound = (iteration - 1) * (self.mini_batch_size * self.inputs_size);
            higher_bound = iteration * self.mini_batch_size * self.inputs_size;
        }
    }
}