// use crate::logic::activations_fns::base_activation_fn::ActivationFN;
// use crate::Arr;

// pub struct init {}

// impl ActivationFN for init {
//     fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
//         z.mapv_inplace(|x| {
//             x.tanh()
//         });
//         z
//     }

//     fn propogate<'a>(&self, z: &'a mut Arr) -> &'a mut Arr {
//         self.forward(z);
//         z.mapv_inplace(|x| {
//             f64::powf((1. - x), 2.)
//         });
//         z
//     }
// }