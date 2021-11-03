use std::collections::HashMap;
use regex::{Regex};
use std::io::{self, BufRead};

use super::utils::LoaderUtils;

struct WordLoader<'a> {
    stop_words: &'a HashMap<&'a str, bool>,
    utils: &'a LoaderUtils,
    file_path: &'a str,
    window_size: usize
}

impl WordLoader<'_> {
    pub fn new<'a>(
        stop_words: &'a HashMap<&'a str, bool>,
        utils: &'a LoaderUtils,
        file_path: &'a str,
        window_size: usize
    ) -> WordLoader<'a> {
        WordLoader {
            stop_words,
            utils,
            file_path,
            window_size
        }
    }

    pub fn load(&self) ->  (HashMap<String, Vec<String>>, usize) {
        let mut corpus = vec!();
        let mut word_to_index: HashMap<String, usize> = HashMap::new();
        let mut index_to_word: HashMap<usize, String> = HashMap::new();
        let mut str = "".to_string();
        for line in self.utils.read_lines(self.file_path) {
            let line = line.unwrap();
            str = format!("{}\n{}", str, line);
        }

        let sentences: Vec<&str> = str.rsplit('.').collect();
        let mut count = 0;

        for sentence in sentences {
            let re = Regex::new(r"[A-Za-z]+").unwrap();
            let mat = re.split(sentence);
            let mut new_line = "".to_string();
            for word in mat {
                if let None = self.stop_words.get(word) {
                    let word = word.to_lowercase();
                    corpus.push(word.clone());
                    if let None = word_to_index.get(&word) {
                        word_to_index.insert(word.clone(), count);
                        index_to_word.insert(count, word.clone());
                        count += 1;
                    }
                }
            }
        }

        let vocab_size= count;
        let corpus_size = corpus.len();
        let mut data: HashMap<String, Vec<String>> = HashMap::new();

        for (i, word) in corpus.iter().enumerate() {
            let index_trg_word = i;
            let mut context_words: Vec<String> = vec!();

            if i == 0 {
                let con = &corpus[1..self.window_size+1];
                con.iter().for_each(|w| context_words.push(w.to_string()));
            } else if i == corpus_size - 1 {
                let con = &corpus[corpus_size-2-self.window_size..corpus_size-2];
                con.iter().rev().for_each(|w| context_words.push(w.to_string()));
            } else {
                let before_trg_wrd_index = index_trg_word - 1;
                (before_trg_wrd_index-self.window_size..before_trg_wrd_index).rev().for_each(|j| {
                    if j < corpus_size {
                        context_words.push(corpus[j].clone());
                    }
                });

                let after_trg_wrd_index = index_trg_word + 1;
                (after_trg_wrd_index..after_trg_wrd_index+self.window_size).rev().for_each(|j| {
                    if j < corpus_size {
                        context_words.push(corpus[j].clone());
                    }
                });
            }

            data.insert(word.to_string(), context_words.iter().map(|c| c.to_string()).collect());
        }

        (data, vocab_size)
    }
}


