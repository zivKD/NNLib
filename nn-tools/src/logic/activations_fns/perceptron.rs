
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::Arr;

pub struct init {}

impl ActivationFN for init {
    fn forward<'a>(&self, z: &'a Arr) -> Arr {
        z.mapv(|x| {
            if(x > 1.) {
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
    const PERCEPTRON: init = init {};

    #[test]
    fn perceptron_forward(){
        let arr : Arr = arr2(&[[1.,0.,0.3], [1.5,2.5,-5.]]);
        let result = arr2(&[
            [0., 0., 0.], 
            [1., 1. , 0.]
        ]);
        assert_eq!(PERCEPTRON.forward(&arr), result);
    }

    // #[test]
    // fn perceptron_propogate(){
    //     let arr : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
    //     let result = arr2(&[
    //         [0.19661193324148185, 0.10499358540350662, 0.045176659730912], 
    //         [0.14914645207033286, 0.07010371654510807, 0.028453023879735598]
    //     ]);
    //     assert_eq!(PERCEPTRON.propogate(&arr), result);
    // }
}