use crate::Arr;

pub trait ActivationFN {
    fn forward<'a>(&self, z: &'a mut Arr) -> &'a mut Arr;
    fn propogate<'a>(&self, z: &'a mut Arr) -> &'a mut Arr;
}
