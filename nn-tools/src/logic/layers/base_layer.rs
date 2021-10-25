use crate::{Arr, ArrView};
pub trait Layer {
    fn feedforward(&mut self, inputs: ArrView) -> Arr;
    fn propogate(&mut self, gradient: Arr, activations: ArrView) -> Arr;
}