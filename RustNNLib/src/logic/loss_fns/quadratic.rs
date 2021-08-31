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
        &a.view() - y
    }
}