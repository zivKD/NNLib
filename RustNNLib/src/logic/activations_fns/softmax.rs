// use crate::logic::activations_fns::base_activation_fn::ActivationFN;
// use crate::Arr;

// pub struct init {}

// impl ActivationFN for init {
//     fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
//         // let max_value = z.map_axis(Axis(0), |view| *view.iter().max().unwrap());
//         // z.mapv_inplace(|x| {
//         //     f64::exp(x) - max_value
//         // });
//         // z
//     }

//     fn propogate<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
//         // self.forward(z);
//         // z.mapv_inplace(|x| 1. - x);
//         // z
//     }
// }