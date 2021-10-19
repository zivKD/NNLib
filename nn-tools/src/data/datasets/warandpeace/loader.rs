use ndarray::{Order, s};
use ndarray::ShapeBuilder; // Needed for .strides() method

use crate::Arr;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;


pub struct Loader<'a> {
    file_path: &'a str,
    train_set_frac: usize,
    test_set_frac: usize,
    validation_set_frac: usize,
    batch_size: usize,
    seq_size: usize
}

impl Loader<'_> {
    pub fn new<'a>(
        file_path: &'a str,
        train_set_frac: usize,
        test_set_frac: usize,
        validation_set_frac: usize,
        batch_size: usize,
        seq_size: usize
    ) -> Loader<'a> {
        Loader {
            file_path,
            train_set_frac,
            test_set_frac,
            validation_set_frac,
            batch_size,
            seq_size
        }
    }


    pub fn build(&self) -> (Arr, Arr, Arr, Arr, Arr, Arr, usize) {
        let mut total_size = 0;
        let mut token_to_index: HashMap<char, usize> = HashMap::new();
        for line in self.read_lines(self.file_path) {
            let line = line.unwrap();
            total_size += line.chars().count();
            for char in line.chars() {
                if let None = token_to_index.get(&char) {
                    token_to_index.insert(char, token_to_index.len() + 1);
                }
            }
        } 

        let word_dim = token_to_index.len() + 1;
        let train_size = (self.train_set_frac as f64 / 100.) * total_size as f64;
        let test_size = (self.test_set_frac as f64 / 100.) * total_size as f64;
        let validation_size = (self.validation_set_frac as f64 / 100.) * total_size as f64;

        let mut train_set = Arr::zeros((train_size as usize, 1));
        let mut test_set = Arr::zeros((test_size as usize, 1));
        let mut validation_set = Arr::zeros((validation_size as usize, 1));

        let splits = [&mut train_set, &mut test_set, &mut validation_set];
        let mut split_idx = 0;
        let mut cur_idx = 0;
        for line in self.read_lines(self.file_path) {
            if split_idx == 3 {
                break;
            }

            for char in line.unwrap().chars() {
                if split_idx == 3 {
                    break;
                }

                let numerical_value = *token_to_index.get(&char).unwrap() as f64;
                splits[split_idx][(cur_idx, 0)] = numerical_value;
                cur_idx += 1;

                if cur_idx == splits[split_idx].dim().0 {
                    split_idx += 1;
                    cur_idx = 0;
                }
            }
        } 

        let (trn_data, trn_lbls) = self.get_ordered_data_and_label_sets(&train_set);
        let (tst_data, tst_lbls) = self.get_ordered_data_and_label_sets(&test_set);
        let (val_data, val_lbls) = self.get_ordered_data_and_label_sets(&validation_set);

        (trn_data, trn_lbls, tst_data, tst_lbls, val_data, val_lbls, word_dim)
    }

    fn get_ordered_data_and_label_sets(&self, set: &Arr) -> (Arr, Arr) {
        let size = set.len();
        let mut cur_size = self.batch_size;
        if self.batch_size * self.seq_size > (size - 1) {
           cur_size = ((size - 1) as f64 / self.seq_size as f64).floor() as usize;
        }

        let mut extra = size % (cur_size * self.seq_size);
        if extra == 0 {
            extra = cur_size * self.seq_size;
        }

        let new_size = size - extra;
        let num_of_rows = new_size / cur_size;
        let data_set = set.slice(s![0..new_size, ..]).to_shape(((num_of_rows, cur_size), Order::ColumnMajor)).unwrap().to_owned();
        let lbl_set = set.slice(s![1..new_size+1, ..]).to_shape(((num_of_rows, cur_size), Order::ColumnMajor)).unwrap().to_owned();
        (data_set, lbl_set)
    }

    fn read_lines<P>(&self, filename: P) -> io::Lines<io::BufReader<File>>
    where P: AsRef<Path>, {
        let file = File::open(filename).unwrap();
        io::BufReader::new(file).lines()
    }
}