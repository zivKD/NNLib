// use crate::logic::activations_fns::base_activation_fn::ActivationFN;
// use crate::Arr;

// pub struct init {}

// impl ActivationFN for init {
//     fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
//         z.mapv_inplace(|x| {
//             f64::max(x, 0.)
//         });
//         z
//     }

//     fn propogate(&self, z: &mut Arr) -> &'a mut Arr {
//         z.mapv_inplace(|x| {
//             (x >= 0.0) as i8 as f64
//         });
//         z
//     }
// }