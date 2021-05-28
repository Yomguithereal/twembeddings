use std::error::Error;
use std::collections::{HashMap, VecDeque, HashSet};
use std::process;

use sparseset::SparseSet;
use serde::Deserialize;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Deserialize)]
struct Record {
    dimensions: String,
    weights: String
}

type SparseVector = Vec<(usize, f64)>;

const LIMIT: usize = 100_000;
// const LIMIT: usize = usize::MAX;

const THRESHOLD: f64 = 0.69;
const WINDOW: usize = 1_500_000;
const TICK: usize = 1_000;
const QUERY_SIZE: u8 = 5;
const VOC_SIZE: usize = 300_000;

fn sparse_dot_product_distance(helper: &mut SparseSet<f64>, first: &SparseVector, second: &SparseVector) -> f64 {
    let mut shortest = first;
    let mut longest = second;

    if first.len() > second.len() {
        shortest = second;
        longest = first;
    }

    let mut product = 0.0;

    for (dim, w1) in shortest.iter() {
        helper.insert(*dim, *w1);
    }

    for (dim, w2) in longest.iter() {
        let w1 = helper.get(*dim);

        if let Some(w1) = w1 {
            product += w1 * w2;
        }
    }

    helper.clear();
    return 1.0 - product;
}

// TODO: vectors deque should suffice with offsets
// TODO: sparse set also for inverted index
// TODO: verify mean/median candidate set size
fn clustering() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("../data/vectors.csv")?;
    let mut i = 0;

    let bar = ProgressBar::new(7_000_000);

    bar.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] < [{eta_precise}] {bar:70} {pos:>7}/{len:7}"));

    let mut cosine_helper_set: SparseSet<f64> = SparseSet::with_capacity(VOC_SIZE);
    let mut inverted_index: HashMap<usize, VecDeque<usize>> = HashMap::new();
    let mut vectors: VecDeque<usize> = VecDeque::new();
    let mut vectors_map: HashMap<usize, SparseVector> = HashMap::new();
    let mut nearest_neighbors: Vec<(usize, f64)> = Vec::new();

    for result in rdr.deserialize() {
        if i >= LIMIT {
            break;
        }

        if i % TICK == 0 {
            bar.inc(TICK as u64);
        }

        let record: Record = result?;

        // println!("{:?}", record);

        let mut sparse_vector: SparseVector = Vec::new();

        if record.dimensions.is_empty() {
            nearest_neighbors.push((i, 0.0));
            i += 1;
            continue;
        }

        let iterator = record.dimensions
            .split("|")
            .zip(record.weights.split("|"));

        for (dimension, weight) in iterator {
            let dimension: usize = dimension.parse()?;
            let weight: f64 = weight.parse()?;
            // println!("Dimension: {:?}, Weight: {:?}", dimension, weight);
            sparse_vector.push((dimension, weight));
            // println!("Vector: {:?}", sparse_vector);
        }

        // Indexing and gathering candidates
        let mut candidates: HashSet<usize> = HashSet::new();
        let mut dim_tested: u8 = 0;

        for (dim, _) in sparse_vector.iter() {
            let deque = inverted_index.entry(*dim).or_default();

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
        let mut best_distance = 2.0;
        let mut best_candidate: Option<usize> = None;

        for candidate in candidates.iter() {
            let other_sparse_vector = vectors_map.get(candidate).unwrap();

            let d = sparse_dot_product_distance(&mut cosine_helper_set, &sparse_vector, other_sparse_vector);

            // println!("{:?}", d);

            if d > THRESHOLD {
                continue;
            }

            if d < best_distance {
                best_distance = d;
                best_candidate = Some(*candidate);
            }
        }

        match best_candidate {
            Some(best_candidate_index) => {
                nearest_neighbors.push((best_candidate_index, best_distance));
            }
            None => {
                nearest_neighbors.push((i, 0.0));
            }
        }

        // Adding tweet to the window
        vectors_map.insert(i, sparse_vector);
        vectors.push_back(i);

        // Dropping tweets from the window
        if vectors_map.len() > WINDOW {
            let to_remove = vectors.pop_front().unwrap();
            let other_sparse_vector = vectors_map.remove(&to_remove).unwrap();

            for (dim, _) in other_sparse_vector.iter() {
                let deque = inverted_index.get_mut(dim).unwrap();
                deque.pop_front().unwrap();
            }
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
