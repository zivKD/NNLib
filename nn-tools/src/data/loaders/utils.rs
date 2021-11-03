use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead};

pub struct LoaderUtils {

}

impl LoaderUtils {
    pub fn read_lines<P>(&self, filename: P) -> io::Lines<io::BufReader<File>>
    where P: AsRef<Path>, {
        let file = File::open(filename).unwrap();
        io::BufReader::new(file).lines()
    }
}