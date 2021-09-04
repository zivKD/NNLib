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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use ndarray::arr2;
    const SIGMOID: init = init {};

    #[test]
    fn sigmoid_forward(){
        let arr : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
        let result = arr2(&[
            [0.7310585786300049, 0.8807970779778823, 0.9525741268224334], 
            [0.8175744761936437, 0.9241418199787566, 0.9706877692486436]
        ]);
        assert_eq!(SIGMOID.forward(&arr), result);
    }

    #[test]
    fn sigmoid_propogate(){
        let arr : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
        let result = arr2(&[
            [0.19661193324148185, 0.10499358540350662, 0.045176659730912], 
            [0.14914645207033286, 0.07010371654510807, 0.028453023879735598]
        ]);
        assert_eq!(SIGMOID.propogate(&arr), result);
    }
}