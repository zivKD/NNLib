pub struct QuadraticLossFN {}

pub trait LossFN {
    fn loss(&self, a: Vec<f64>, y: Vec<f64>) -> Vec<f64>;
    fn derivative(&self, a: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Vec<f64>;
}

mod math {
    pub fn forLoop<F>(len: usize, func: F)
        where F: FnMut(usize) -> () {
        let mut i = 0;
        while i < len {
            func(i);
            i = i + 1;
        }
    }

    // fn vectorAction<F>(len: usize, func: F)
    //     where F: Fn(Vec<f64>) -> Fn(usize) -> () {
    //     let mut c = Vec::new();
    //     forLoop(len, func(c));
    //     return c;
    // }

    pub fn vectorSubraction(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
        let mut c : Vec<f64> = (0f64..a.len() as f64).collect();
        forLoop(a.len(), |i: usize| {
            c[i] = a[i] - b[i];
        });

        return c;
    }

    pub fn vectorPower(a: Vec<f64>, power: f64) -> Vec<f64> {
        let mut c = Vec::new();
        forLoop(a.len(), |i: usize| {
            c[i] = a[i].powf(power);
        });

        return c;
    }

    pub fn vectorScalarMutliplication(a: Vec<f64>, scalar: f64) -> Vec<f64> {
        let mut c = Vec::new();

        forLoop(a.len(), |i: usize| {
            c[i] = a[i] * scalar;
        });

        return c;
    }
}

impl LossFN for QuadraticLossFN {
    fn loss(&self, a: Vec<f64>, y: Vec<f64>) -> Vec<f64> {
        let v1 = math::vectorSubraction(a, y);     
        let v2 = math::vectorPower(v1, 2f64);     
        return math::vectorScalarMutliplication(v2, 0.5f64);
    }

    fn derivative(&self, a: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Vec<f64> {
       return vec!(10f64, 10f64, 10f64);
    }
}

fn main() {
    let quatratic = QuadraticLossFN {};
    let v = quatratic.loss(vec!(10f64, 10f64, 10f64), vec!(8f64, 8f64, 8f64));
    println!("Initial vector: {:?}", v);
}