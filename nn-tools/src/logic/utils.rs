use ndarray_stats::QuantileExt;
use rand::thread_rng;
use ndarray::{Array1, Array2, ArrayView1, Order, Shape, ShapeBuilder};
use ndarray::{Axis, Zip};
use ndarray_rand::RandomExt;
use rand::prelude::SliceRandom;
use crate::Arr;
use crate::logic::activations_fns::base_activation_fn::ActivationFN;

pub fn round_decimal(places: u32, x: f64) -> f64 {
    let percision = i32::pow(10, places) as f64;
    (x * percision).round() / percision
}

pub fn repeat(get_fn: &dyn Fn((usize, usize),) -> f64, desired_shape: &(usize, usize)) -> Arr {
    // Probably not cost affective, should improve performenece
    Arr::from_shape_fn(*desired_shape, get_fn)
}

pub fn repeated_axis_zero(arr: &Arr, desired_shape: &(usize, usize)) -> Arr {
    // Probably not cost affective, should improve performenece
    repeat(&|(i, _j)| *arr.get((i, 0)).unwrap(), desired_shape)
}

pub fn arr_zeros_with_shape(shape: &[usize]) -> Arr {
    Arr::zeros((shape[0], shape[1]))
}

pub fn arr_ones_with_shape(shape: &[usize]) -> Arr {
    Arr::ones((shape[0], shape[1]))
}

pub fn one_hot_encoding(a: &Arr, word_dim: usize) -> Arr {
    let rows = a.shape()[0];
    let columns = a.shape()[1];
    let identity_matrix = Arr::eye(word_dim);
    let mut one_hot_encoded: Vec<f64> = Vec::new();

    let inv_shape = &[a.shape()[1], a.shape()[0]];
    iterate_throgh_2d(inv_shape, |(i, j)| {
        let mut row = identity_matrix.row(a[(j,i)] as usize).to_vec();
        one_hot_encoded.append(&mut row); 
    });

    Arr::from_shape_vec((word_dim * rows, columns).strides((1, word_dim * rows)), one_hot_encoded).unwrap()
}

pub fn gradient_clipping(gradient: &Arr, min_value: f64, max_value: f64) -> Arr {
    let mut gradient_clone = gradient.clone();
    iterate_throgh_2d(gradient.shape(), |(i, j)| {
        let f = gradient[(i,j)];
        if f > max_value { 
            gradient_clone[(i,j)] = max_value;
        } else if f < min_value {
            gradient_clone[(i,j)] = min_value;
        }
    });

    gradient_clone
}

fn iterate_throgh_2d<T: FnMut((usize, usize))>(shape: &[usize], mut action: T) {
    (0..shape[0]).for_each(|i| {
        (0..shape[1]).for_each(|j| {
            action((i, j));
        })
    });
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arr;
    use ndarray::arr2;

    // #[test]
    // fn cross_softmax_success(){
    //     // self.word_dim X miniBatchSize
    //     let mut arr : Arr = arr2(&[[0.21, 0.02], [0.45, 0.73], [0.34, 0.25]]);
    //     // 1 X miniBatchSize
    //     let labels : Arr = arr2(&[[2., 1.]]);
    //     let result = arr2(&[[0.21, 0.02], [0.45, -0.27], [-0.66, 0.25]]);
    //     cross_entropy_propogate(&mut arr, &labels);
    //     assert_eq!(result, arr);
    // }

    #[test]
    fn one_hot_encoding_success(){
        let arr : Arr = arr2(&[[1.,3.,5.], [0., 3., 2.]]);
        let result = arr2(&[
            [0., 0. , 0.], 
            [1., 0. , 0.], 
            [0., 0. , 0.], 
            [0., 1. , 0.], 
            [0., 0. , 0.], 
            [0., 0. , 1.], 
            [1., 0. , 0.], 
            [0., 0. , 0.], 
            [0., 0. , 1.], 
            [0., 1. , 0.], 
            [0., 0. , 0.], 
            [0., 0. , 0.], 
        ]);
        assert_eq!(result, one_hot_encoding(&arr, 6));
    }

    #[test]
    fn gradient_clipping_success(){
        let arr : Arr = arr2(&[[1.,2.,3.], [1.5,2.5,3.5]]);
        let max_value = 3.;
        let min_value = 2.;
        let result = arr2(&[
            [2., 2., 3.], 
            [2., 2.5, 3.]
        ]);
        assert_eq!(gradient_clipping(&arr, min_value, max_value), result);
    }
}