// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025-2026 TensorChord Inc.

mod emeans;
mod flat;
mod hierarchical;
mod index;
mod quick;
mod rabitq;

pub mod square;

pub use emeans::e_means;

use crate::square::{Square, SquareMut};
use rand::rngs::StdRng;

pub struct This<'a> {
    pool: &'a rayon::ThreadPool,
    rng: StdRng,
    d: usize,
    c: usize,
    is_spherical: bool,
}

pub trait KMeans {
    fn prefect_index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_>;
    fn index(&self) -> Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> {
        self.prefect_index()
    }
    fn assign(&mut self);
    fn update(&mut self);
    fn finish(self: Box<Self>) -> Square;
}

pub fn k_means_lookup(sample: &[f32], centroids: &Square) -> usize {
    use simd::Floating;
    let mut result = (f32::INFINITY, 0);
    for (i, centroid) in centroids.into_iter().enumerate() {
        let dis = f32::reduce_sum_of_d2(sample, centroid);
        if dis <= result.0 {
            result = (dis, i);
        }
    }
    result.1
}

pub fn lloyd_k_means<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    assert!(d > 0 && c > 0);
    let n = samples.len();
    if n <= c {
        quick::new(pool, d, samples, c, seed, is_spherical)
    } else if n <= c * 2 {
        flat::new(pool, d, samples, c, seed, is_spherical)
    } else {
        rabitq::new(pool, d, samples, c, seed, is_spherical)
    }
}

pub fn hierarchical_k_means<'a>(
    pool: &'a rayon::ThreadPool,
    d: usize,
    samples: SquareMut<'a>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
) -> Box<dyn KMeans + 'a> {
    assert!(d > 0 && c > 0);
    hierarchical::new(pool, d, samples, c, seed, is_spherical)
}
