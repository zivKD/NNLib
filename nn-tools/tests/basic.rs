// use nnTools::Arr;
// use nnTools::logic::loss_fns::base_loss_fn::LossFN;
// use nnTools::logic::layers::base_layer::Layer;
// use nnTools::logic::activations_fns::perceptron;
// use nnTools::logic::gradient_decents::stochastic;
// use nnTools::logic::layers::fully_connected;
// use nnTools::logic::loss_fns::quadratic;
// use ndarray::{ arr2 };
// use rand::Rng;


// #[test]
// fn basic_perceptron_success() {
//     // This tests checks we can teach a single perceptron to return 1 if a number is bigger then X, or 0 if it's smaller
//     const X: f64 = 5.;
//     let perceptron_fn = perceptron::init {};
//     let weights = &mut arr2(&[[1.]]);
//     let biases = &mut arr2(&[[1.]]);
//     let stochastic = stochastic::init::new(0.03, 10);
//     let mut one_neuron = fully_connected::init::new(
//         1,
//         1,
//         weights,
//         biases,
//         &perceptron_fn,
//         &stochastic,
//     );
//     let quadratic = quadratic::init {};
//     let mut avarage_error : f64 = 1.;
//     let mut counter = 0;
//     let mut good_result_counter = 0;
//     let mut rng = rand::thread_rng();
//     while good_result_counter < 10 {
//         let num = rng.gen_range(0.0..10.0);
//         let input : Arr = arr2(&[[num]]);
//         let expected_output: Arr = if num > X {
//             arr2(&[[1.]])
//         } else {
//             arr2(&[[0.]])
//         };

//         // println!("input: {:?}", input);
//         // println!("expected_output: {:?}", expected_output);
//         let mut output = one_neuron.feedforward(input);
//         let error = quadratic.propogate(&mut output, &expected_output);
//         avarage_error = error.mapv(|x| {if x < 0. { return -x} else { return x }}).sum() / error.len() as f64;
//         // println!("network error: {:?}", &avarage_error);
//         one_neuron.propogate(error);

//         if counter < 6 {
//             counter+=1;
//         } else {
//             counter = 0;
//         }

//         if avarage_error < 0.1 {
//             good_result_counter+=1;
//         } else {
//             good_result_counter=0;
//         }
//     }

//     assert_eq!(good_result_counter, 10)
// }