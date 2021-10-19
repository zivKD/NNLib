use ndarray::Zip;
use ndarray::ShapeBuilder; // Needed for .strides() method
use nntools::Arr;
use mnist::MnistBuilder;
use ndarray_stats::QuantileExt;
use nntools::data::datasets::{self};

#[test]
fn mnist_loader(){
    let mnist_loader = datasets::mnist::loader::Loader::new(
        "./src/data/datasets/mnist/files",
        "train-labels.idx1-ubyte",
        "train-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        50000 as u32,
        10000 as u32,
        10000 as u32,
    );

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
    let (
        trn_img, 
        trn_lbl, 
        tst_img, 
        tst_lbl, 
        val_img, 
        val_lbl
    ) = mnist_loader.build(); 

    let first_image_by_mnist = (0..1000).map(|i| *mnist.trn_img.get(i).unwrap() as f64/ 256.).collect::<Vec<f64>>();
    let first_image_by_loader = (0..1000).map(|i| *trn_img.get((i, 0)).unwrap()).collect::<Vec<f64>>();
    let first_lbls_by_mnist = (0..1000).map(|i| *mnist.trn_lbl.get(i).unwrap() as usize).collect::<Vec<usize>>();
    let first_lbls_by_loader = (0..1000).map(|i| trn_lbl.column(i).argmax().unwrap()).collect::<Vec<usize>>(); 
    assert_eq!(first_image_by_mnist, first_image_by_loader);
    assert_eq!(first_lbls_by_mnist, first_lbls_by_loader); 

    let first_image_by_mnist = (0..1000).map(|i| *mnist.tst_img.get(i).unwrap()  as f64 / 256.).collect::<Vec<f64>>();
    let first_image_by_loader = (0..1000).map(|i| *tst_img.get((i, 0)).unwrap()).collect::<Vec<f64>>();
    let first_lbls_by_mnist = (0..1000).map(|i| *mnist.tst_lbl.get(i).unwrap() as usize).collect::<Vec<usize>>();
    let first_lbls_by_loader = (0..1000).map(|i| tst_lbl.column(i).argmax().unwrap()).collect::<Vec<usize>>(); 
    assert_eq!(first_image_by_mnist, first_image_by_loader);
    assert_eq!(first_lbls_by_mnist, first_lbls_by_loader);  

    let first_image_by_mnist = (0..1000).map(|i| *mnist.val_img.get(i).unwrap() as f64/ 256.).collect::<Vec<f64>>();
    let first_image_by_loader = (0..1000).map(|i| *val_img.get((i, 0)).unwrap()).collect::<Vec<f64>>();
    let first_lbls_by_mnist = (0..1000).map(|i| *mnist.val_lbl.get(i).unwrap() as usize).collect::<Vec<usize>>();
    let first_lbls_by_loader = (0..1000).map(|i| val_lbl.column(i).argmax().unwrap()).collect::<Vec<usize>>(); 
    assert_eq!(first_image_by_mnist, first_image_by_loader);
    assert_eq!(first_lbls_by_mnist, first_lbls_by_loader); 
} 

#[test]
fn warandpeace_loader() {
    let loader = datasets::warandpeace::loader::Loader::new(
        "./src/data/datasets/warandpeace/files/tst.txt",
        50,
        25,
        25,
        2,
        5
    );

    let expected_trn_data = vec!(1., 2., 3., 2., 4., 2., 5., 2., 6., 2., 7., 2., 8., 2., 9., 2., 10., 2., 11., 2.);
    let expected_trn_data = Arr::from_shape_vec((10, 2).strides((1, 10)), expected_trn_data).unwrap();
    let expected_trn_lbls = vec!(2., 3., 2., 4., 2., 5., 2., 6., 2., 7., 2., 8., 2., 9., 2., 10., 2., 11., 2., 12.);
    let expected_trn_lbls = Arr::from_shape_vec((10, 2).strides((1,10)), expected_trn_lbls).unwrap();
    let expected_tst_data = vec!(15., 2., 16., 2., 17., 2., 18., 2., 19., 2.);
    let expected_tst_data = Arr::from_shape_vec((5, 2).strides((1,5)), expected_tst_data).unwrap();
    let expected_tst_lbls = vec!(2., 16., 2., 17., 2., 18., 2., 19., 2., 20.);
    let expected_tst_lbls = Arr::from_shape_vec((5, 2).strides((1,5)), expected_tst_lbls).unwrap();
    let expected_val_data = vec!(2., 22., 2., 23., 2., 24., 2., 25., 2., 26.);
    let expected_val_data = Arr::from_shape_vec((5, 2).strides((1,5)), expected_val_data).unwrap();
    let expected_val_lbls = vec!(22., 2., 23., 2., 24., 2., 25., 2., 26., 2.);
    let expected_val_lbls = Arr::from_shape_vec((5, 2).strides((1,5)), expected_val_lbls).unwrap();
    let (
        trn_data, 
        trn_lbls,
        tst_data,
        tst_lbls,
        val_data,
        val_lbls,
        word_dim
    ) = loader.build();

    assert_eq!(29, word_dim);
    assert_eq(trn_data, expected_trn_data, "tr-data");
    assert_eq(trn_lbls, expected_trn_lbls, "trn-lbls");
    assert_eq(tst_data, expected_tst_data, "tst-data");
    assert_eq(tst_lbls, expected_tst_lbls, "tst-lbls");
    assert_eq(val_data, expected_val_data, "val-data");
    assert_eq(val_lbls, expected_val_lbls, "val-lbls");
} 

fn assert_eq(set1: Arr, set2: Arr, name: &str) {
    let num_of_equals = Zip::from(&set1).and(&set2).map_collect(|x, y| *x == *y).iter().filter(|b| **b).collect::<Vec<&bool>>().len();
    assert_eq!(set1.len(), num_of_equals, "assert {} is equal", name);
}
