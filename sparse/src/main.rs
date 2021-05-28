use std::error::Error;
use std::collections::HashMap;
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

fn clustering() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("../data/vectors.csv")?;
    let mut i = 0;

    let bar = ProgressBar::new(7_000_000);

    for result in rdr.deserialize() {
        if i >= LIMIT {
            break;
        }
        if i % 10_000 == 0 {
            bar.inc(10_000);
        }
        i += 1;

        let record: Record = result?;

        // println!("{:?}", record);

        let mut sparse_vector: HashMap<u32, f64> = HashMap::new();

        if record.dimensions.is_empty() {
            continue;
        }

        let iterator = record.dimensions
            .split("|")
            .zip(record.weights.split("|"));

        for (dimension, weight) in iterator {
            let dimension: u32 = dimension.parse()?;
            let weight: f64 = weight.parse()?;
            // println!("Dimension: {:?}, Weight: {:?}", dimension, weight);
            sparse_vector.insert(dimension, weight);
            // println!("Vector: {:?}", sparse_vector);
        }
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
