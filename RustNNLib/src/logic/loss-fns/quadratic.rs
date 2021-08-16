use main::LossFN;
use main::math::vectorPower;
use main::math::vectorScalarMutliplication;
use main::math::vectorSubraction;

pub struct QuadraticLossFN {}

pub impl LossFN for QuadraticLossFN {
    fn loss(&self, a: Vec<f64>, y: Vec<f64>) -> Vec<f64> {
        Vec<f64> v1 = vectorSubraction(a, y);     
        Vec<f64> v2 = vectorPower(v1, 2);     
        return vectorScalarMutliplication(v2, 0.5);
    }

    fn derivative(&self, a: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Vec<f64> {
        
    }
}