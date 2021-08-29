use ndarray::{ Array2 };

pub type Arr = Array2<f64>;

pub trait activation_fn {
    fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr;
    fn propogate<'a>(&self, z: &'a mut Arr) -> &'a mut Arr;
}
