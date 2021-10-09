use std::cell::RefMut;

use crate::{Arr, logic::{activations_fns::base_activation_fn::ActivationFN, layers::base_layer::Layer, loss_fns::base_loss_fn::LossFN}};

pub struct Network<'a> {
    data_set : &'a Arr,
    lbl_set: &'a Arr,
    mini_batch_size: usize,
    inputs_size: usize,
    data_set_size: usize,
    layers: RefMut<'a, Vec<&'a mut dyn Layer>>,
    loss_fn: &'a dyn LossFN,
    activation_fn: &'a dyn ActivationFN
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
        activation_fn: &'a dyn ActivationFN
    ) -> Network<'a> {
        Network {
            data_set,
            lbl_set,
            mini_batch_size,
            inputs_size,
            data_set_size,
            layers,
            loss_fn,
            activation_fn
        }
    }

    pub fn run(&mut self, print_result: bool) {
        
    }
}