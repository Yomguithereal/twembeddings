use std::error::Error;
use std::collections::{HashMap, VecDeque, HashSet};
use std::process;

use serde::Deserialize;
use indicatif::ProgressBar;

#[derive(Debug, Deserialize)]
struct Record {
    dimensions: String,
    weights: String
}

const LIMIT: usize = 10_000;
// const LIMIT: usize = usize::MAX;

fn sparse_dot_product_distance(first: &HashMap<u32, f64>, second: &HashMap<u32, f64>) -> f64 {
    let mut shortest = first;
    let mut longest = second;

    if first.len() > second.len() {
        shortest = second;
        longest = first;
    }

    let mut product = 0.0;

    for (dim, w1) in shortest.iter() {
        let w2 = longest.get(dim);

        if let Some(w2) = w2 {
            product += w1 * w2;
        }
    }

    return 1.0 - product;
}

fn clustering() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("../data/vectors.csv")?;
    let mut i = 0;

    let bar = ProgressBar::new(7_000_000);

    let mut index: HashMap<u32, VecDeque<usize>> = HashMap::new();

    for result in rdr.deserialize() {
        if i >= LIMIT {
            break;
        }
        i += 1;
        if i % 10_000 == 0 {
            bar.inc(10_000);
        }

        let record: Record = result?;

        // println!("{:?}", record);

        let mut sparse_vector: HashMap<u32, f64> = HashMap::new();

        if record.dimensions.is_empty() {
            continue;
        }

        let iterator = record.dimensions
            .split("|")
            .zip(record.weights.split("|"));

        let mut dimensions: Vec<u32> = Vec::new();

        for (dimension, weight) in iterator {
            let dimension: u32 = dimension.parse()?;
            let weight: f64 = weight.parse()?;
            // println!("Dimension: {:?}, Weight: {:?}", dimension, weight);
            dimensions.push(dimension);
            sparse_vector.insert(dimension, weight);
            // println!("Vector: {:?}", sparse_vector);
        }

        // Indexing and gathering candidates
        let mut candidates: HashSet<usize> = HashSet::new();

        for dim in dimensions {

            // TODO: only query first 5 dimensions later
            let deque = index.entry(dim).or_default();

            for candidate in deque.iter() {
                candidates.insert(*candidate);
            }

            deque.push_back(i);
        }

        // println!("{:?}", candidates.len());
    }
    bar.finish();
    Ok(())
}

fn main() {
    if let Err(err) = clustering() {
        println!("error running clustering: {}", err);
        process::exit(1);
    }
}
