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

type SparseVector = HashMap<u32, f64>;

const LIMIT: usize = 10_000;
// const LIMIT: usize = usize::MAX;

const THRESHOLD: f64 = 0.69;
const WINDOW: usize = 1_500_000;

fn sparse_dot_product_distance(first: &SparseVector, second: &SparseVector) -> f64 {
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

fn clustering<'a>() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("../data/vectors.csv")?;
    let mut i = 0;

    let bar = ProgressBar::new(7_000_000);

    let mut index: HashMap<u32, VecDeque<usize>> = HashMap::new();
    let mut vectors: HashMap<usize, SparseVector> = HashMap::new();
    let mut nearest_neighbors: Vec<(usize, f64)> = Vec::new();

    for result in rdr.deserialize() {
        if i >= LIMIT {
            break;
        }

        if i % 10_000 == 0 {
            bar.inc(10_000);
        }

        let record: Record = result?;

        // println!("{:?}", record);

        let mut sparse_vector: SparseVector = HashMap::new();

        if record.dimensions.is_empty() {
            nearest_neighbors.push((i, 0.0));
            i += 1;
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

        // Finding the nearest neighbor
        let mut best_distance = 2.0;
        let mut best_candidate: Option<usize> = None;

        for candidate in candidates.iter() {
            let other_sparse_vector = vectors.get(candidate).unwrap();

            let d = sparse_dot_product_distance(&sparse_vector, other_sparse_vector);

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

        vectors.insert(i, sparse_vector.to_owned());

        // T.append(t1)

        // if len(T) > window:
        //     t3 = T.popleft()

        //     for dim in t3['vector'].keys():
        //         I[dim].popleft()
        i += 1;
    }

    bar.finish();

    let with_nearest_neighbor: Vec<(usize, f64)> = nearest_neighbors
        .into_iter()
        .enumerate()
        .filter(|(j, c)| j != &c.0)
        .map(|(_, c)| c)
        .collect();

    println!("{:?}, {:?}", with_nearest_neighbor, with_nearest_neighbor.len());

    Ok(())
}

fn main() {
    if let Err(err) = clustering() {
        println!("error running clustering: {}", err);
        process::exit(1);
    }
}
