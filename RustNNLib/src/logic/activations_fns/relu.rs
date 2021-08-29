use crate::logic::activations_fns::base_activation_fn::activation_fn;
use crate::logic::activations_fns::base_activation_fn::Arr;

pub struct init {}

impl activation_fn for init {
    fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
        z.mapv_inplace(|x| {
            f64::max(x, 0.)
        });
        z
    }

    fn propogate<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
        z.mapv_inplace(|x| {
            (x >= 0.0) as i8 as f64
        });
        z
    }
}