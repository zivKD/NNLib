pub trait LossFN {
    fn loss(&self, a: Vec<f64>, y: Vec<f64>) -> Vec<f64>;
    fn derivative(&self, a: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Vec<f64>;
}
