use ndarray_stats::QuantileExt;
use rand::thread_rng;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray::{Axis, Zip};
use ndarray_rand::RandomExt;
use rand::prelude::SliceRandom;
use crate::Arr;

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

pub fn softmax(arr: Arr) -> Arr {
    let max_value = arr.get(arr.argmax().unwrap()).unwrap();
    let mut exps = arr.map(|x| (x-max_value).exp());
    exps.axis_iter_mut(Axis(0)).for_each(|mut axis| {
        let sum = axis.sum();
        axis.mapv_inplace(|x| x/sum);
    });
    exps
}


// NOT WORKING
// pub fn shuffle_sets(data_set: &Arr, lbl_set: &Arr, inputs_size: usize, set_size: usize) -> (Arr, Arr) {
//     let reshaped_data_set = data_set.to_shape((inputs_size, set_size)).unwrap();
//     let mut zip: Vec<(ArrayView1<f64>, ArrayView1<f64>)> = Zip::from(reshaped_data_set.columns()).and(lbl_set.columns()).
//             map_collect(|c1:  ArrayView1<f64>, c2: ArrayView1<f64>| (c1, c2)).to_vec();
//     zip.shuffle(&mut thread_rng());
//     let mut new_vec: Vec<f64> = vec!();
//     zip.iter().for_each(|x| {
//         let mut vec = x.0.to_vec();
//         new_vec.append(&mut vec)
//     });
//     let new_data_set = Arr::from_shape_vec((set_size*inputs_size, 1), new_vec).unwrap();
//     let mut new_vec: Vec<f64> = vec!();
//     zip.iter().for_each(|x| {
//         let mut vec = x.1.to_vec();
//         new_vec.append(&mut vec)
//     });
//     let new_lbls_set = Arr::from_shape_vec((2, set_size), new_vec).unwrap();
//     (new_data_set, new_lbls_set)
// }

// #[cfg(test)]
// mod tests {
//     use std::collections::HashMap;

//     use ndarray_rand::rand_distr::Uniform;

//     use super::*;
//     use crate::Arr;

//     #[test]
//     fn shuffle_correct(){
//         // trn_size = 4, inputs_size = 3
//         let data_set = Arr::random((12, 1), Uniform::new(0.,10.));
//         let lbl_set = Arr::random((2, 4), Uniform::new(0.,10.));
//         let (new_data_set, new_lbl_set) = shuffle_sets(&data_set, &lbl_set, 3, 4);
//         println!("lbl: {:?} new lbl: {:?}", lbl_set, new_lbl_set);
//         // println!("trn: {:?} new trn: {:?}", data_set, new_data_set);
//         assert_eq!(data_set.shape(), new_data_set.shape());
//         assert_eq!(lbl_set.shape(), new_lbl_set.shape());
//         let mut i = 0;
//         let mut j = 0;
//         let mut indices_match: HashMap<i32, i32> = HashMap::new();
//         for r1 in data_set.rows() {
//             for r2 in new_data_set.rows() {
//                 if r2.len() == Zip::from(&r1).and(&r2).map_collect(|f1, f2| f1 == f2).iter().filter(|b| **b).collect::<Vec<&bool>>().len() {
//                     indices_match.insert(i, j);
//                 }

//                 j+=1;
//             }

//             j=0;
//             i+=1;
//         }

//         assert_eq!(indices_match.len(), 12);

//         let mut i = 0;
//         let mut j = 0;
//         let mut indices_match: HashMap<i32, i32> = HashMap::new();
//         for r1 in lbl_set.rows() {
//             for r2 in new_lbl_set.rows() {
//                 if r2.len() == Zip::from(&r1).and(&r2).map_collect(|f1, f2| f1 == f2).iter().filter(|b| **b).collect::<Vec<&bool>>().len() {
//                     indices_match.insert(i, j);
//                 }

//                 j+=1;
//             }

//             j=0;
//             i+=1;
//         }

//         assert_eq!(indices_match.len(), 2);
//     }
// }