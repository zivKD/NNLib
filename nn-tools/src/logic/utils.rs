use ndarray::{Array1, Array2, ArrayView1};
use ndarray::{Axis, Zip};
use ndarray_rand::RandomExt;
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

//TODO
pub fn shuffle_rows(data_set: &Arr, lbl_set: &Arr, inputs_size: usize, set_size: usize) -> (Arr, Arr) {
    // data_set trn_size*inputs_size X 1
    // lbl_set 10 X trn_size
    // inputs_size <-> 10
    // Vec<Vec<(input, output)>> trn_sizeX(10, inputs_size)
    println!("got here 1");
    let reshaped_data_set = data_set.to_shape((inputs_size, set_size)).unwrap();
    println!("got here 2");
    let zip: Array1<(ArrayView1<f64>, ArrayView1<f64>)> = Zip::from(reshaped_data_set.columns()).
    and(lbl_set.columns()).map_collect(|c1:  ArrayView1<f64>, c2: ArrayView1<f64>| (c1, c2));
    println!("zip shape {:?}", zip.shape());
    let shuffled: Array1<(ArrayView1<f64>, ArrayView1<f64>)> = 
        zip.sample_axis(Axis(0), zip.len_of(Axis(0)), ndarray_rand::SamplingStrategy::WithReplacement);
    println!("got here 4");
    let new_data_set: Arr = Arr::from_shape_fn((inputs_size * set_size, 1), |(i, j)| {
        let index_set = i % set_size;
        let index_inputs = i % inputs_size;
        *shuffled.get(index_set).expect(&format!("i: {} set index: {} index index: {}", i, index_set, index_inputs)).
        0.get(index_inputs).expect(&format!("i: {} set index: {} index index: {}", i, index_set, index_inputs))
    });
    println!("got here 5");
    let new_lbl_set: Arr = Arr::from_shape_fn((10, set_size), |(i, j)| {
        let index = (j / set_size) * set_size + j % set_size;
        shuffled.get(index).unwrap().1[i]
    });
    println!("got here 6");

    (new_data_set, new_lbl_set)
}
#[cfg(test)]
mod tests {
    use ndarray_rand::rand_distr::Uniform;

    use super::*;
    use crate::Arr;

    #[test]
    fn shuffle_correct(){
        // trn_size = 8, inputs_size = 4
        let data_set = Arr::random((32, 1), Uniform::new(0.,10.));
        let lbl_set = Arr::random((10, 8), Uniform::new(0.,10.));
        let (new_data_set, new_lbl_set) = shuffle_rows(&data_set, &lbl_set, 4, 8);
        assert_eq!(data_set.shape(), new_data_set.shape());
        println!("got here 7");
        assert_eq!(lbl_set.shape(), new_lbl_set.shape());
        println!("got here 8");
        let row = data_set.row(0);
        let mut new_row_index = 0;
        for new_row in new_data_set.rows() {
            let equality = Zip::from(&new_row).and(&row).map_collect(|x, y| x == y).iter().filter(|b| **b).collect::<Vec<&bool>>().len();
            if equality == new_row.len() {
                break;
            }
            new_row_index+=1;
        }

        println!("got here 9");
        let lbl_row = lbl_set.index_axis(Axis(0), new_row_index % 10);
        println!("got here 10");
        let mut new_row_index_lbl = 0;
        for new_row in new_lbl_set.rows() {
            let equality = Zip::from(&new_row).and(&lbl_row).map_collect(|x, y| x == y).iter().filter(|b| **b).collect::<Vec<&bool>>().len();
            if equality == new_row.len() {
                break;
            }

            new_row_index_lbl+=1;
        }
        println!("original {:?} new {:?}", data_set, new_data_set);
        assert_eq!(new_row_index % 10, new_row_index_lbl);
    }
}