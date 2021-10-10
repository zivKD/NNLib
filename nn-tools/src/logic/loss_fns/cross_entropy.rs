use ndarray::{Axis, Zip};
use crate::{Arr, DEFAULT};
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::loss_fns::base_loss_fn::LossFN; 

pub struct Init<'a> {
    activation_fn: &'a dyn ActivationFN
}

impl Init<'_> {
    pub fn new<'a>(
        activation_fn: &'a dyn ActivationFN
    ) -> Init {
        Init {
            activation_fn
        }
    }
}
impl LossFN for Init<'_> {
    fn output<'a>(&self, a: &'a mut Arr, y: &'a Arr) -> Arr {
        let loss = Zip::from(a).and(y).map_collect(|a_x, y_x| y_x * a_x.log(2.));
        loss.map_axis(Axis(1), |axis| -axis.sum()).into_shape((loss.shape()[0], 1)).unwrap()
    }

    fn propogate<'a>(&self,z: &'a mut Arr, a: &'a mut Arr, y: &'a Arr) -> Arr {
        DEFAULT()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use crate::logic::activations_fns::sigmoid;
    use crate::logic::utils::round_decimal;
    use ndarray::arr2;
    const SIGMOID: sigmoid::Init = sigmoid::Init {};

    #[test]
    fn correct_output(){
        let cross_entropy: Init = Init::new(&SIGMOID);
        let mut a : Arr = arr2(&[[0.2, 0.1, 0.7], [0.123, 0.407, 0.48]]);
        let y: Arr = arr2(&[[1., 0., 0.], [0., 0., 1.]]);
        let result: Arr = arr2(&[[2.321928], [1.058894]]);
        assert_eq!(cross_entropy.output(&mut a, &y).mapv(|x| round_decimal(6, x)), result);
    }
}