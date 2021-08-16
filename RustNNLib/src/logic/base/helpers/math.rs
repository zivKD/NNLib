pub fn vectorSubraction(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let mut c : Vec<f64> =  vec![f64, a.len()];
    for (i,x) in a.iter().enumarate() {
        c[i] = x - b[i];
    }

    return c;
}

pub fn vectorPower(a: Vec<f64>, power: i32) -> Vec<f64> {
    let mut c : Vec<f64> = a;
    for (i,x) in c.iter().enumarte() {
        c[i] = f64::pow(x, power);
    }

    return c;
}

pub fn vectorScalarMutliplication(a: Vec<f64>, scalar: f64) {
    let mut c : Vec<f64> = a;
    for (i,x) in c.iter().enumarte() {
        c[i] = x * scalar;
    }

    return c;
}