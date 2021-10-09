use ndarray_stats::QuantileExt;

use ndarray::Axis;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::{Arr, DEFAULT};

pub struct Init {}

impl ActivationFN for Init {
    fn forward<'a>(&self, z: &'a Arr) -> Arr {
        let max_value = z.get(z.argmax().unwrap()).unwrap();
        let mut exps = z.map(|x| (x-max_value).exp());
        exps.axis_iter_mut(Axis(0)).for_each(|mut axis| {
            let sum = axis.sum();
            axis.mapv_inplace(|x| x/sum);
        });
        exps
    }

    fn propogate<'a>(&self, z: &'a Arr) -> Arr {
        DEFAULT()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arr, logic::utils::round_decimal};
    use ndarray::arr2;
    const SOFTMAX: Init = Init {};

    #[test]
    fn softmax_forward(){
        let arr : Arr = arr2(&[[1.,2.,3.], [0.2, 0.5, 0.8]]);
        let result = arr2(&[
            [0.090030573, 0.244728471, 0.665240956], 
            [0.239694479, 0.323553704, 0.436751817]
        ]);
        assert_eq!(SOFTMAX.forward(&arr).map(|x| round_decimal(9, *x)), result);
    }
}