use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;

pub struct Init {}

impl ActivationFN for Init {
    fn forward<'a>(&self, z: &'a Arr) -> Arr {
        z.mapv(|x| {
            if x > 1. {
                1.
            } else {
                0.
            }
        })
    }

    fn propogate<'a>(&self, z: &'a Arr) -> Arr {
        self.forward(z).mapv(|x| {
            (1. - x) * x
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use ndarray::arr2;
    const PERCEPTRON: Init = Init {};

    #[test]
    fn perceptron_forward(){
        let arr : Arr = arr2(&[[1.,0.,0.3], [1.5,2.5,-5.]]);
        let result = arr2(&[
            [0., 0., 0.], 
            [1., 1. , 0.]
        ]);
        assert_eq!(PERCEPTRON.forward(&arr), result);
    }
}