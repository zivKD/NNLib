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
        let val_lbl = self.get_lbl_set(val_lbl, val_size);


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
        Arr::from_shape_fn((10, lbls_size), |(i, j)| {
            let mut val = 0.;
            if *lbls_set.get((j, 0)).unwrap() as usize == i {
                val = 1.;
            }

            val
        })
    }
}

#[cfg(test)]
mod tests {
    use mnist::Mnist;
    use ndarray_stats::QuantileExt;

    use super::*;
    // intg test
    #[test]
    fn first_image_is_identical(){
        let (trn_size, tst_size, val_size, rows, cols) = (50_000 as usize, 10_000 as usize, 10_000 as usize, 28 as usize, 28 as usize);
        let mnist = MnistBuilder::new()
            .base_path("./src/data/datasets/mnist/files")
            .training_images_filename("train-images.idx3-ubyte")
            .training_labels_filename("train-labels.idx1-ubyte")
            .test_images_filename("t10k-images.idx3-ubyte")
            .test_labels_filename("t10k-labels.idx1-ubyte")
            .label_format_digit()
            .training_set_length(trn_size as u32)
            .validation_set_length(val_size as u32)
            .test_set_length(tst_size as u32)
            .finalize();

        let mnist_loader = Loader::new(
            "./src/data/datasets/mnist/files",
            "train-labels.idx1-ubyte",
            "train-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte",
            "t10k-images.idx3-ubyte",
            trn_size as u32,
            tst_size as u32,
            val_size as u32,
            rows,
            cols
        );

        let (
            trn_img, 
            trn_lbl, 
            tst_img, 
            tst_lbl, 
            val_img, 
            val_lbl
        ) = mnist_loader.build();

        let first_image_by_mnist = (0..1000).map(|i| *mnist.trn_img.get(i).unwrap() as f64).collect::<Vec<f64>>();
        let first_image_by_loader = (0..1000).map(|i| *trn_img.get((i, 0)).unwrap()).collect::<Vec<f64>>();
        let first_lbls_by_mnist = (0..1000).map(|i| *mnist.trn_lbl.get(i).unwrap() as usize).collect::<Vec<usize>>();
        let first_lbls_by_loader = (0..1000).map(|i| trn_lbl.column(i).argmax().unwrap()).collect::<Vec<usize>>();

        assert_eq!(first_image_by_mnist, first_image_by_loader);
        assert_eq!(first_lbls_by_mnist, first_lbls_by_loader);

        let first_image_by_mnist = (0..1000).map(|i| *mnist.tst_img.get(i).unwrap() as f64).collect::<Vec<f64>>();
        let first_image_by_loader = (0..1000).map(|i| *tst_img.get((i, 0)).unwrap()).collect::<Vec<f64>>();
        let first_lbls_by_mnist = (0..1000).map(|i| *mnist.tst_lbl.get(i).unwrap() as usize).collect::<Vec<usize>>();
        let first_lbls_by_loader = (0..1000).map(|i| tst_lbl.column(i).argmax().unwrap()).collect::<Vec<usize>>();

        assert_eq!(first_image_by_mnist, first_image_by_loader);
        assert_eq!(first_lbls_by_mnist, first_lbls_by_loader);


        let first_image_by_mnist = (0..1000).map(|i| *mnist.val_img.get(i).unwrap() as f64).collect::<Vec<f64>>();
        let first_image_by_loader = (0..1000).map(|i| *val_img.get((i, 0)).unwrap()).collect::<Vec<f64>>();
        let first_lbls_by_mnist = (0..1000).map(|i| *mnist.val_lbl.get(i).unwrap() as usize).collect::<Vec<usize>>();
        let first_lbls_by_loader = (0..1000).map(|i| val_lbl.column(i).argmax().unwrap()).collect::<Vec<usize>>();

        assert_eq!(first_image_by_mnist, first_image_by_loader);
        assert_eq!(first_lbls_by_mnist, first_lbls_by_loader);
    }
}