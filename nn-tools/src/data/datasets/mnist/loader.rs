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

        let trn_img = self.get_img_set(mnist.trn_img);
        let trn_lbl : Arr = self.get_lbl_set(mnist.trn_lbl);
        let tst_img = self.get_img_set(mnist.tst_img);
        let tst_lbl : Arr = self.get_lbl_set(mnist.tst_lbl);
        let val_img = self.get_img_set(mnist.val_img);
        let val_lbl : Arr = self.get_lbl_set(mnist.val_lbl);

        return (trn_img, trn_lbl, tst_img, tst_lbl, val_img, val_lbl);
    }

    fn get_img_set(&self, set: Vec<u8>) -> Arr {
        let vec_f64: Vec<f64> = set.iter().map(|x| *x as f64 / 256.).collect();
        Arr::from_shape_vec((vec_f64.len(), 1), vec_f64).unwrap()
    }

    fn get_lbl_set(&self, set: Vec<u8>) -> Arr {
        let vec_f64: Vec<f64> = set.iter().map(|x| *x as f64).collect();
        /* Return a 10-dimensional unit vector with a 1.0 in the jth
            position and zeroes elsewhere.  This is used to convert a digit
            (0...9) into a corresponding desired output from the neural
            network.
        */
        Arr::from_shape_fn((10, vec_f64.len()), |(i, j)| {
            let mut val = 0.;
            if *vec_f64.get(j).unwrap() as usize == i {
                val = 1.;
            }

            val
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::logic::utils::round_decimal;

    use super::*;

    #[test]
    fn first_image_is_identical() {
        let mnist_loader: Loader = Loader::new(
            "./src/data/datasets/mnist/files",
            "train-labels.idx1-ubyte",
            "train-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte",
            "t10k-images.idx3-ubyte",
            50000 as u32,
            10000 as u32,
            10000 as u32,
        );

        let (trn_size, tst_size, val_size, _, _) = (50_000 as usize, 10_000 as usize, 10_000 as usize, 28 as usize, 28 as usize);
        let first_img: Vec<f64> = vec!(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01171875, 0.0703125 , 0.0703125 , 0.0703125 , 0.4921875 , 0.53125   , 0.68359375, 0.1015625 , 0.6484375 , 0.99609375, 0.96484375, 0.49609375, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.1171875 , 0.140625  , 0.3671875 , 0.6015625 , 0.6640625 , 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.87890625, 0.671875  , 0.98828125, 0.9453125 , 0.76171875, 0.25      , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.19140625, 0.9296875 , 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98046875, 0.36328125, 0.3203125 , 0.3203125 , 0.21875   , 0.15234375, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.0703125 , 0.85546875, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.7734375 , 0.7109375 , 0.96484375, 0.94140625, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.3125    , 0.609375  , 0.41796875, 0.98828125, 0.98828125, 0.80078125, 0.04296875, 0.        , 0.16796875, 0.6015625 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.0546875 , 0.00390625, 0.6015625 , 0.98828125, 0.3515625 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.54296875, 0.98828125, 0.7421875 , 0.0078125 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.04296875, 0.7421875 , 0.98828125, 0.2734375 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.13671875, 0.94140625, 0.87890625, 0.625     , 0.421875  , 0.00390625, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.31640625, 0.9375    , 0.98828125, 0.98828125, 0.46484375, 0.09765625, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.17578125, 0.7265625 , 0.98828125, 0.98828125, 0.5859375 , 0.10546875, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.0625    , 0.36328125, 0.984375  , 0.98828125, 0.73046875, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.97265625, 0.98828125, 0.97265625, 0.25      , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.1796875 , 0.5078125 , 0.71484375, 0.98828125, 0.98828125, 0.80859375, 0.0078125 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.15234375, 0.578125  , 0.89453125, 0.98828125, 0.98828125, 0.98828125, 0.9765625 , 0.7109375 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.09375   , 0.4453125 , 0.86328125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.78515625, 0.3046875 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.08984375, 0.2578125 , 0.83203125, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.7734375 , 0.31640625, 0.0078125 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.0703125 , 0.66796875, 0.85546875, 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.76171875, 0.3125    , 0.03515625, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.21484375, 0.671875  , 0.8828125 , 0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.953125, 0.51953125, 0.04296875, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.53125   , 0.98828125, 0.98828125, 0.98828125, 0.828125  , 0.52734375, 0.515625  , 0.0625    , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
        let first_img: Vec<f64> = first_img.iter().map(|x| round_decimal(6, *x)).collect();
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
        
        let trn_set = mnist_loader.get_img_set(mnist.trn_img);
        let first_image_by_loader = (0..784).map(|i| *trn_set.get((i, 0)).unwrap()).collect::<Vec<f64>>();
        let first_image_by_loader: Vec<f64> = first_image_by_loader.iter().map(|x| round_decimal(6, *x)).collect();
        assert_eq!(first_img, first_image_by_loader);
    }
}