use crate::Arr;

pub trait ActivationFN {
    fn forward<'a>(&self, z: &'a Arr) -> Arr;
    fn propogate<'a>(&self, z: &'a Arr) -> Arr;
}
