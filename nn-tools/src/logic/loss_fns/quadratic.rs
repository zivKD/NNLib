use crate::Arr;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::loss_fns::base_loss_fn::LossFN; 

pub struct Init<'a> {
    activation_fn: &'a dyn ActivationFN
}

impl Init<'_> {
    pub fn new<'a>(activation_fn: &'a dyn ActivationFN) -> Init {
        Init {
            activation_fn: activation_fn
        }
    }
}

impl LossFN for Init<'_> {
    fn output<'a>(&self, a: &'a Arr, y: &'a Arr) -> Arr {
       let mut pos  = &a.view() - y;
       pos.map(|x| {
           0.5 * f64::powf(*x, 2.)
       })
    }

    fn propogate<'a>(&self,z: &'a Arr, a: &'a Arr, y: &'a Arr) -> Arr {
        // // δl=δl+1⊙σ′(zl)
        (&a.view() - y) * self.activation_fn.propogate(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use crate::logic::utils::round_decimal;
    use crate::logic::activations_fns::sigmoid;
    use ndarray::arr2;

    #[test]
    fn correct_output(){
        let QUADRATIC: Init = Init::new(&sigmoid::Init{});
        let mut a : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
        let y: Arr = arr2(&[[0.8, 0.6, 0.5], [0.2, 0.4, 0.2]]);
        let result: Arr = arr2(&[[0.02, 0.002312, 0.049298], [0.00608305, 0.00060552, 0.0032]]);
        assert_eq!(QUADRATIC.output(&mut a, &y).mapv(|x| round_decimal(8, x)), result);
    }

    #[test]
    fn correct_propogate() {
        let QUADRATIC: Init = Init::new(&sigmoid::Init{});
        let mut a : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
        let mut z : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
        let y: Arr = arr2(&[[0.8, 0.6, 0.5], [0.2, 0.4, 0.2]]);
        let a_minus_y: Arr = arr2(&[[0.2, -0.068, 0.314], [0.1103, 0.0348, -0.08]]);
        let activation_derivative = arr2(&[
            [0.19661193324148185, 0.10499358540350662, 0.045176659730912], 
            [0.14914645207033286, 0.07010371654510807, 0.028453023879735598]
        ]);
        assert_eq!(QUADRATIC.propogate(&mut z, &mut a, &y).mapv(|x| round_decimal(6, x)), (a_minus_y * activation_derivative).mapv(|x| round_decimal(6, x)));
    }
}