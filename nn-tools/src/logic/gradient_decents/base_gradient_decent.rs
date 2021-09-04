use crate::Arr;

pub trait GradientDecent {
    fn change_weights<'a>(&self, w: &'a mut Arr, gradient: & Arr);
    fn change_biases<'a>(&self, b: &'a mut Arr, gradient: & Arr);
}
