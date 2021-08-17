pub fn vector_subtraction(a:&Vec<f64>, b:&Vec<f64>) -> Vec<f64> {
    let mut c = Vec::new();
    for (i, &_x) in a.iter().enumerate() {
        c.push(a[i] - b[i]);
    }

    c
}

pub fn vector_power(a: &Vec<f64>, power: f64) -> Vec<f64> {
    let mut c = Vec::new();
    for x in a.iter() {
        c.push(f64::powf(*x, power));
    }

    c
}

pub fn vector_scalar_multiplication(a: &Vec<f64>, scalar: f64) -> Vec<f64> {
    let mut c : Vec<f64> = Vec::new();
    for x in a.iter() {
        c.push(*x * scalar);
    }

    c
}