use crate::Arr;
use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;

pub struct init {}

impl GradientDecent for init {
    fn change_weights<'a>(&self, w: &'a Arr, gradient: &'a Arr) -> Arr {
        let rate = 0.03 / 10.;
        let balanced_gradient = rate * gradient;
        w - balanced_gradient 
    }

    fn change_biases<'a>(&self, b: &'a Arr, gradient: &'a Arr) -> Arr {
        let rate = 0.03 / 10.;
        let balanced_gradient = rate * gradient;
        b - balanced_gradient 
    }
}