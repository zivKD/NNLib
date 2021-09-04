use crate::Arr;
use crate::logic::loss_fns::base_loss_fn::LossFN; 

pub struct init {}

impl LossFN for init {
    fn output<'a>(&self, a: &'a mut Arr, y: &'a Arr) -> Arr {
       let mut pos  = &a.view() - y;
       pos.mapv_inplace(|x| {
           0.5 * f64::powf(x, 2.)
       });

       pos 
    }

    fn propogate<'a>(&self, a: &'a mut Arr, y: &'a Arr) -> Arr {
        y - &a.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use crate::logic::utils::round_decimal;
    use ndarray::arr2;
    const QUADRATIC: init = init {};

    #[test]
    fn correct_output(){
        let mut a : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
        let y: Arr = arr2(&[[0.8, 0.6, 0.5], [0.2, 0.4, 0.2]]);
        let result: Arr = arr2(&[[0.02, 0.002312, 0.049298], [0.00608305, 0.00060552, 0.0032]]);
        assert_eq!(QUADRATIC.output(&mut a, &y).mapv(|x| round_decimal(8, x)), result);
    }

    #[test]
    fn correct_propogate() {
        let mut a : Arr = arr2(&[[1.0, 0.532, 0.814], [0.3103, 0.4348, 0.12]]);
        let y: Arr = arr2(&[[0.8, 0.6, 0.5], [0.2, 0.4, 0.2]]);
        let result: Arr = arr2(&[[-0.2, 0.068, -0.314], [-0.1103, -0.0348, 0.08]]);
        assert_eq!(QUADRATIC.propogate(&mut a, &y).mapv(|x| round_decimal(4, x)), result);
    }
}