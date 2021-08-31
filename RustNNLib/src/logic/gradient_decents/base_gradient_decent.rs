use crate::Arr;

pub trait GradientDecent {
    fn change_weights<'a>(&self, w: &'a Arr, gradient: &'a Arr) -> Arr;
    fn change_biases<'a>(&self, b: &'a Arr, gradient: &'a Arr) -> Arr;
}
