use crate::Arr;

pub fn round_decimal(places: u32, x: f64) -> f64 {
    let percision = i32::pow(10, places) as f64;
    (x * percision).round() / percision
}

pub fn repeat(get_fn: &dyn Fn((usize, usize),) -> f64, desired_shape: &(usize, usize)) -> Arr {
        Arr::from_shape_fn(*desired_shape, get_fn)
}

pub fn repeated_axis_zero(arr: &Arr, desired_shape: &(usize, usize)) -> Arr {
    repeat(&|(i, j)| *arr.get((i, 0)).unwrap(), desired_shape)
}