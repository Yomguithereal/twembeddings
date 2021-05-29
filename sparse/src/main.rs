use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::process;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use sparseset::SparseSet;

#[derive(Debug, Deserialize)]
struct CSVRecord {
    dimensions: String,
    weights: String,
}

type SparseVector = Vec<(usize, f64)>;

// const LIMIT: usize = 10_000;
const LIMIT: usize = usize::MAX;

const THRESHOLD: f64 = 0.69;
const WINDOW: usize = 1_500_000;
const TICK: usize = 1_000;
const QUERY_SIZE: u8 = 5;
const VOC_SIZE: usize = 300_000;

fn sparse_dot_product_distance(helper: &SparseSet<f64>, other: &SparseVector) -> f64 {
    let mut product = 0.0;

    for (dim, w2) in other {
        let w1 = helper.get(*dim).unwrap_or(&0.0);
        product += w1 * w2;
    }

    product = 1.0 - product;

    // Precision error
    if product < 0.0 {
        return 0.0;
    }

    product
}

// TODO: verify mean/median candidate set size
// TODO: use a max clamp for normalize and for distance
// TODO: candidate set can also be a sparse set?
// TODO: sanity tests
// TODO: canopy + canopy variant where dim has exactly same value?
// TODO: start from window directly to easy test
// TODO: try clippy
// TODO: sparse set with array
// TODO: try inlining
// TODO: remove mentions from tokenization, try an upper df filter on proportion?
// TODO: try f32
fn clustering() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("../data/vectors.csv")?;
    let mut i = 0;
    let mut dropped_so_far = 0;

    let bar = ProgressBar::new(7_000_000);

    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] < [{eta_precise}] {bar:70} {pos:>7}/{len:7}"),
    );

    let mut cosine_helper_set: SparseSet<f64> = SparseSet::with_capacity(VOC_SIZE);
    let mut inverted_index: SparseSet<VecDeque<usize>> = SparseSet::with_capacity(VOC_SIZE);
    let mut vectors: VecDeque<SparseVector> = VecDeque::new();
    let mut nearest_neighbors: Vec<(usize, f64)> = Vec::new();
    let mut candidates: HashSet<usize> = HashSet::with_capacity(10_000);

    for result in rdr.deserialize() {
        if i >= LIMIT {
            break;
        }

        if i % TICK == 0 {
            bar.inc(TICK as u64);
        }

        let record: CSVRecord = result?;

        // println!("{:?}", record);

        let mut sparse_vector: SparseVector = Vec::new();

        if record.dimensions.is_empty() {
            nearest_neighbors.push((i, 0.0));
            i += 1;
            vectors.push_back(sparse_vector);
            continue;
        }

        cosine_helper_set.clear();

        let iterator = record.dimensions.split("|").zip(record.weights.split("|"));

        for (dimension, weight) in iterator {
            let dimension: usize = dimension.parse()?;
            let weight: f64 = weight.parse()?;
            // println!("Dimension: {:?}, Weight: {:?}", dimension, weight);
            sparse_vector.push((dimension, weight));
            // println!("Vector: {:?}", sparse_vector);
            cosine_helper_set.insert(dimension, weight);
        }

        // Indexing and gathering candidates
        let mut dim_tested: u8 = 0;

        for (dim, _) in sparse_vector.iter() {
            // TODO: do better
            if !inverted_index.contains(*dim) {
                inverted_index.insert(*dim, VecDeque::new());
            }

            let deque = inverted_index.get_mut(*dim).unwrap();

            if dim_tested < QUERY_SIZE {
                for candidate in deque.iter() {
                    candidates.insert(*candidate);
                }
                dim_tested += 1;
            }

            deque.push_back(i);
        }

        // println!("{:?}", candidates.len());

        // Finding the nearest neighbor
        let best: Option<(usize, f64)> = candidates
            .par_iter()
            .map(|candidate| {
                let other_sparse_vector = &vectors[*candidate - dropped_so_far];
                return (
                    *candidate,
                    sparse_dot_product_distance(&cosine_helper_set, &other_sparse_vector),
                );
            })
            .filter(|x| x.1 < THRESHOLD)
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

        candidates.clear();

        match best {
            Some((c, d)) => {
                nearest_neighbors.push((c, d));
            }
            None => {
                nearest_neighbors.push((i, 0.0));
            }
        }

        // Adding tweet to the window
        vectors.push_back(sparse_vector);

        // Dropping tweets from the window
        if vectors.len() > WINDOW {
            let to_remove = vectors.pop_front().unwrap();

            for (dim, _) in to_remove.iter() {
                let deque = inverted_index.get_mut(*dim).unwrap();
                deque.pop_front().unwrap();
            }

            dropped_so_far += 1;
        }

        i += 1;
    }

    bar.finish();

    // println!("{:?}", nearest_neighbors.len());

    // let with_nearest_neighbor: Vec<(usize, f64)> = nearest_neighbors
    //     .into_iter()
    //     .enumerate()
    //     .filter(|(j, c)| j != &c.0)
    //     .map(|(_, c)| c)
    //     .collect();

    // println!("{:?}, {:?}", with_nearest_neighbor, with_nearest_neighbor.len());

    Ok(())
}

fn main() {
    if let Err(err) = clustering() {
        println!("error running clustering: {}", err);
        process::exit(1);
    }
}
