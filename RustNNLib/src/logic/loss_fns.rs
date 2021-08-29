// use super::base:: { helpers as helpers, LossFN as LossFN };

// pub struct QuadraticLossFN {}

// impl LossFN for QuadraticLossFN {
//     fn loss(&self, a: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
//         let v1 = helpers::vector_subtraction(a, y);     
//         let v2 = helpers::vector_power(&v1, 2.0);     
//         helpers::vector_scalar_multiplication(&v2, 0.5)
//     }

//     fn derivative(&self, a: &Vec<f64>, y: &Vec<f64>, z: &Vec<f64>) -> Vec<f64> { 
//         assert_eq!(a.len(), y.len());
//         let sub = helpers::vector_subtraction(a, y);
//         todo!();
//         // self.activation.derivative(sub) * sub
//     }
// }