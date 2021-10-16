use ndarray::{Axis, Zip};
use ndarray_stats::QuantileExt;
use crate::{Arr, DEFAULT};
use crate::logic::activations_fns::base_activation_fn::ActivationFN;
use crate::logic::loss_fns::base_loss_fn::LossFN; 

pub struct Init {
}

impl Init {
    fn softmax_forward(&self, z: &Arr) -> Arr {
        let max_value = z.get(z.argmax().unwrap()).unwrap();
        let mut exps = z.map(|x| (x-max_value).exp());
        exps.axis_iter_mut(Axis(0)).for_each(|mut axis| {
            let sum = axis.sum();
            axis.mapv_inplace(|x| x/sum);
        });
        exps
    }
}

impl LossFN for Init {
    fn output<'a>(&self, a: &'a Arr, y: &'a Arr) -> Arr {
        let loss = Zip::from(a).and(y).map_collect(|a_x, y_x| y_x * a_x.log(2.));
        loss.map_axis(Axis(1), |axis| -axis.sum()).into_shape((loss.shape()[0], 1)).unwrap()
    }

    fn propogate<'a>(&self,z: &'a Arr, a: &'a Arr, y: &'a Arr) -> Arr {
        let mut probs = self.softmax_forward(a);
        y.columns().into_iter().enumerate().for_each(|(i, c)| c.for_each(|f| probs[(*f as usize, i)] -=1.));
        probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use crate::logic::utils::round_decimal;
    use ndarray::arr2;
    const CROSS_ENTROPY: Init = Init {};

    #[test]
    fn correct_output(){
        let mut a : Arr = arr2(&[[0.2, 0.1, 0.7], [0.123, 0.407, 0.48]]);
        let y: Arr = arr2(&[[1., 0., 0.], [0., 0., 1.]]);
        let result: Arr = arr2(&[[2.321928], [1.058894]]);
        assert_eq!(CROSS_ENTROPY.output(&mut a, &y).mapv(|x| round_decimal(6, x)), result);
    }

    #[test]
    fn softmax_forward(){
        let arr : Arr = arr2(&[[1.,2.,3.], [0.2, 0.5, 0.8]]);
        let result = arr2(&[
            [0.090030573, 0.244728471, 0.665240956], 
            [0.239694479, 0.323553704, 0.436751817]
        ]);
        assert_eq!(CROSS_ENTROPY.softmax_forward(&arr).map(|x| round_decimal(9, *x)), result);
    }
}