pub mod logic;
pub mod data;

use ndarray::ArrayView2;
use ndarray::Array2;

pub type Arr = Array2<f64>;
pub type ArrView<'a> = ArrayView2<'a, f64>;
pub fn default_arr_value() -> Arr { Arr::default((1,1)) }