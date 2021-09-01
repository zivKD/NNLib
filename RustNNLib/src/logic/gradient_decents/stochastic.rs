use crate::Arr;
use crate::logic::gradient_decents::base_gradient_decent::GradientDecent;
use ndarray::Zip;

pub struct init {}

impl GradientDecent for init {
    fn change_weights<'a>(&self, w: &'a mut Arr, gradient: & Arr) {
        let rate = 0.03 / 10.;
        let balanced_gradient = rate * gradient;
        Zip::from(w).and(&balanced_gradient).for_each(|x, &y| *x -= y);
    }

    fn change_biases<'a>(&self, b: &'a mut Arr, gradient: & Arr) {
        let rate = 0.03 / 10.;
        let balanced_gradient = rate * gradient;
        Zip::from(b).and(&balanced_gradient).for_each(|x, &y| *x -= y);
    }
}