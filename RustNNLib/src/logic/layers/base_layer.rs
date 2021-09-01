use crate::Arr;

pub trait Layer {
    fn feedforward(&mut self, inputs: Arr) -> Arr;
    fn propogate(&mut self, gradient: Arr) -> Arr;
}