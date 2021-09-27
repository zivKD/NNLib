pub struct Loader {
    base_path: &str,
    train_set_lbls_filename: &str,
    train_set_data_filename: &str,
    test_set_lbls_filename: &str,
    test_set_data_filename: &str,
    train_set_size: u32,
    test_set_size: u32,
    validation_set_size: u32,
    input_rows_size: usize,
    input_cols_size: usize
}

impl Loader {
    pub fn new(
        base_path: &str,
        train_set_lbls_filename: &str,
        train_set_data_filename: &str,
        test_set_lbls_filename: &str,
        test_set_data_filename: &str,
        train_set_size: u32,
        test_set_size: u32,
        validation_set_size: u32,
        input_rows_size: usize,
        input_cols_size: usize
    ) -> Load {
        Load {
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


    pub fn build(&self) {
        let mnist = MnistBuilder::new()
            .base_path(self.base_path)
            .training_images_filename(self.train_set_lbls_filename)
            .training_labels_filename(self.train_set_data_filename)
            .test_images_filename(self.test_set_lbls_filename)
            .test_labels_filename(self.test_set_data_filename)
            .label_format_digit()
            .training_set_length(self.train_set_size)
            .validation_set_length(self.validation_set_size)
            .test_set_length(self.test_set_size)
            .finalize();

        let trn_img_f64 : Vec<f64> = mnist.trn_img.iter().map(|x| *x as f64).collect();
        trn_img = Arr::from_shape_vec((trn_size*rows*cols, 1), trn_img_f64).unwrap();
        let trn_lbl = Arr::from_shape_vec((trn_size, 1), trn_lbl_f64).unwrap();
        /* Return a 10-dimensional unit vector with a 1.0 in the jth
            position and zeroes elsewhere.  This is used to convert a digit
            (0...9) into a corresponding desired output from the neural
            network.
        */
        // let tst_img_f64 : Vec<f64> = mnist.tst_img.iter().map(|x| *x as f64).collect();
        // let tst_lbl_f64 : Vec<f64> = mnist.tst_lbl.iter().map(|x| *x as f64).collect();
        // let tst_img = Arr::from_shape_vec((tst_size*rows*cols, 1), tst_img_f64).unwrap();
        // let tst_lbl = Arr::from_shape_vec((tst_size, 1), tst_lbl_f64).unwrap();

        let val_img_f64 : Vec<f64> = mnist.val_img.iter().map(|x| *x as f64).collect();
        let val_lbl_f64 : Vec<f64> = mnist.val_lbl.iter().map(|x| *x as f64).collect();
        let val_img = Arr::from_shape_vec((val_size*rows*cols, 1), val_img_f64).unwrap();
        let val_lbl = Arr::from_shape_vec((val_size, 1), val_lbl_f64).unwrap();
    }

    fn get_lbl_set(lbls_set: Arr, lbls_size: u32) {
        let new_lbls_set = Arr::from_shape_fn((lbls_size , 10), |(i, j)| {
            let mut val = 0.;
            if *trn_lbl.get((i, 0)).unwrap() as usize == j {
                val = 1.;
            }

            val
        });
    }
}