use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;

pub struct Init {}

impl ActivationFN for Init {
    fn forward<'a>(&self, z: &'a Arr) -> Arr {
        z.map(|x| {
            x.tanh()
        })
    }

    fn propogate<'a>(&self, z: &'a Arr) -> Arr {
        self.forward(z);
        z.map(|x| {
            f64::powf((1. - x), 2.)
        })
    }
}