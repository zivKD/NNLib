use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;

pub struct init {}

impl ActivationFN for init {
    fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
        z.mapv_inplace(|x| {
            1.0 / (1.0 + f64::exp(-x))
        });
        z
    }

    fn propogate<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
        self.forward(z);
        z.mapv_inplace(|x| {
            (1. - x) * x
        });
        z
    }
}