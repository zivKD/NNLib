use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;

pub struct init {}

impl ActivationFN for init {
    fn forward<'a>(&self, z: &'a Arr) -> Arr {
        z.mapv(|x| {
            1.0 / (1.0 + f64::exp(-x))
        })
    }

    fn propogate<'a>(&self, z: &'a Arr) -> Arr {
        self.forward(z).mapv(|x| {
            (1. - x) * x
        })
    }
}