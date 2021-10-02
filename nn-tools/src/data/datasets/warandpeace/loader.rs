use mnist::MnistBuilder;

use crate::Arr;

pub struct Loader<'a> {
    file_path: &'a str,
    train_set_lbls_filename: &'a str,
    train_set_data_filename: &'a str,
    test_set_lbls_filename: &'a str,
    test_set_data_filename: &'a str,
    train_set_size: u32,
    test_set_size: u32,
    validation_set_size: u32,
}

impl Loader<'_> {
    pub fn new<'a>(
        file_path: &'a str,
        train_set_lbls_filename: &'a str,
        train_set_data_filename: &'a str,
        test_set_lbls_filename: &'a str,
        test_set_data_filename: &'a str,
        train_set_size: u32,
        test_set_size: u32,
        validation_set_size: u32,
    ) -> Loader<'a> {
        Loader {
            file_path,
            train_set_lbls_filename,
            train_set_data_filename,
            test_set_lbls_filename,
            test_set_data_filename,
            train_set_size,
            test_set_size,
            validation_set_size,
        }
    }


    pub fn build(&self) {
    }
}

#[cfg(test)]
mod tests {
    use ndarray_stats::QuantileExt;
    use crate::logic::utils::round_decimal;

    use super::*;

    // intg test
    #[test]
    fn rwos_are_identical(){
    } 
}