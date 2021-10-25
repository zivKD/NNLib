use crate::Arr;
pub trait LossFN {
    fn output<'a>(&self, a: &'a Arr, y: &'a Arr) -> Arr;
    fn propogate<'a>(&self,z: &'a Arr, a: &'a Arr, y: &'a Arr) -> Arr;
}
