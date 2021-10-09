use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;

pub struct Init {}

impl ActivationFN for Init {
    fn forward<'a>(&self, z: &'a Arr) -> Arr {
        z.map(|x| x.tanh())
    }

    fn propogate<'a>(&self, z: &'a Arr) -> Arr {
        let z = self.forward(z);
        z.map(|x| {
            1. - x.powf(2.)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arr, logic::utils::round_decimal};
    use ndarray::arr2;
    const TANH: Init = Init {};

    #[test]
    fn tanh_forward(){
        let arr : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
        let result = arr2(&[
            [0.761594, 0.964028, 0.995055], 
            [0.905148, 0.986614, 0.998178]
        ]);
        assert_eq!(TANH.forward(&arr).mapv(|x| round_decimal(6, x)), result);
    }

    #[test]
    fn tanh_propogate(){
        let arr : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
        let result = arr2(&[
            [0.419974, 0.070651, 0.009866], 
            [0.180707, 0.026592, 0.003641]
        ]);
        assert_eq!(TANH.propogate(&arr).mapv(|x| round_decimal(6, x)), result);
    }
}