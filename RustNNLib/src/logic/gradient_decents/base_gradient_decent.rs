use crate::Arr;

pub trait LossFN {
    fn output<'a>(&self, a: &'a mut Arr, y: &'a Arr) -> Arr;
    fn propogate<'a>(&self, a: &'a mut Arr, y: &'a Arr) -> Arr;
}
