use mnist::MnistBuilder;

use crate::Arr;

pub struct Loader<'a> {
    base_path: &'a str,
    train_set_lbls_filename: &'a str,
    train_set_data_filename: &'a str,
    test_set_lbls_filename: &'a str,
    test_set_data_filename: &'a str,
    train_set_size: u32,
    test_set_size: u32,
    validation_set_size: u32,
    input_rows_size: usize,
    input_cols_size: usize
}

impl Loader<'_> {
    pub fn new<'a>(
        base_path: &'a str,
        train_set_lbls_filename: &'a str,
        train_set_data_filename: &'a str,
        test_set_lbls_filename: &'a str,
        test_set_data_filename: &'a str,
        train_set_size: u32,
        test_set_size: u32,
        validation_set_size: u32,
        input_rows_size: usize,
        input_cols_size: usize
    ) -> Loader<'a> {
        Loader {
            base_path,
            train_set_lbls_filename,
            train_set_data_filename,
            test_set_lbls_filename,
            test_set_data_filename,
            train_set_size,
            test_set_size,
            validation_set_size,
            input_rows_size,
            input_cols_size
        }
    }


    pub fn build(&self) -> (Arr, Arr, Arr, Arr, Arr, Arr) {
        let mnist = MnistBuilder::new()
            .base_path(&self.base_path)
            .training_images_filename(&self.train_set_data_filename)
            .training_labels_filename(&self.train_set_lbls_filename)
            .test_images_filename(&self.test_set_data_filename)
            .test_labels_filename(&self.test_set_lbls_filename)
            .label_format_digit()
            .training_set_length(self.train_set_size)
            .validation_set_length(self.validation_set_size)
            .test_set_length(self.test_set_size)
            .finalize();

        let trn_size = self.train_set_size as usize;
        let trn_img_f64 : Vec<f64> = self.get_vectorized_set(mnist.trn_img);
        let trn_lbl_f64 : Vec<f64> = self.get_vectorized_set(mnist.trn_lbl);
        let trn_img = Arr::from_shape_vec((trn_size*self.input_rows_size*self.input_cols_size, 1), trn_img_f64).unwrap();
        let trn_lbl = Arr::from_shape_vec((trn_size, 1), trn_lbl_f64).unwrap();
        let trn_lbl =  self.get_lbl_set(trn_lbl, trn_size);

        let tst_size = self.test_set_size as usize;
        let tst_img_f64 : Vec<f64> = self.get_vectorized_set(mnist.tst_img);
        let tst_lbl_f64 : Vec<f64> = self.get_vectorized_set(mnist.tst_lbl);
        let tst_img = Arr::from_shape_vec((tst_size*self.input_rows_size*self.input_cols_size, 1), tst_img_f64).unwrap();
        let tst_lbl = Arr::from_shape_vec((tst_size, 1), tst_lbl_f64).unwrap();
        let tst_lbl =  self.get_lbl_set(tst_lbl, tst_size);

        let val_size = self.validation_set_size as usize;
        let val_img_f64 : Vec<f64> = self.get_vectorized_set(mnist.val_img);
        let val_lbl_f64 : Vec<f64> = self.get_vectorized_set(mnist.val_lbl);
        let val_img = Arr::from_shape_vec((val_size*self.input_rows_size*self.input_cols_size, 1), val_img_f64).unwrap();
        let val_lbl = Arr::from_shape_vec((val_size, 1), val_lbl_f64).unwrap();


        return (trn_img, trn_lbl, tst_img, tst_lbl, val_img, val_lbl);
    }

    fn get_vectorized_set(&self, set: Vec<u8>) -> Vec<f64> {
        set.iter().map(|x| *x as f64).collect()
    }

    fn get_lbl_set(&self, lbls_set: Arr, lbls_size: usize) -> Arr {
        /* Return a 10-dimensional unit vector with a 1.0 in the jth
            position and zeroes elsewhere.  This is used to convert a digit
            (0...9) into a corresponding desired output from the neural
            network.
        */
        Arr::from_shape_fn((lbls_size , 10), |(i, j)| {
            let mut val = 0.;
            if *lbls_set.get((i, 0)).unwrap() as usize == j {
                val = 1.;
            }

            val
        })
    }
}