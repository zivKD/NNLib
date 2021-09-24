use crate::{Arr, ArrViewMut};

pub trait Layer {
    fn feedforward(&mut self, inputs: ArrViewMut) -> Arr;
    fn propogate(&mut self, gradient: Arr) -> Arr;
}