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
// The recipe and optimizer are adapted from Faiss 1.16.0 NumpyEmeans:
// https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means
// Bessa et al., e-means: outperforming k-means (in one-tenth the epochs)
// https://arxiv.org/abs/2611.09501
//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// MIT License
// Copyright (c) Facebook, Inc. and its affiliates.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::index::{flat_index, rabitq_index};
use crate::square::{Square, SquareMut};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use simd::Floating;

const INITIAL_USAGE: f32 = 1.0 / 65_536.0;
const REVIVAL_INTERVAL: usize = 5;
const UNUSED_THRESHOLD: f32 = 0.1;
const SPLIT_EPSILON: f32 = 1e-4;

struct ClusterStats {
    members: Vec<usize>,
    usage: f32,
    closing_count: usize,
}

/// Train e-means centroids for the given number of full passes over the sample.
/// Uses VectorChord's RaBitQ assignment for larger samples, rotating their
/// workspace in place and returning the centroids in the original coordinates.
/// `on_batch` receives the zero-based epoch on the calling thread before each update.
#[allow(clippy::too_many_arguments)]
pub fn e_means(
    pool: &rayon::ThreadPool,
    d: usize,
    mut samples: SquareMut<'_>,
    c: usize,
    seed: [u8; 32],
    is_spherical: bool,
    epochs: usize,
    mut on_batch: impl FnMut(usize),
) -> Square {
    assert!(d > 0 && c > 0 && epochs > 0);
    assert_eq!(d, samples.d());
    let n = samples.len();
    let use_rabitq = c > 1 && n > c.saturating_mul(2);
    if use_rabitq {
        pool.install(|| {
            samples
                .par_iter_mut()
                .for_each(rabitq::rotate::rotate_inplace)
        });
    }

    let mut init_rng = StdRng::from_seed(seed);
    let mut centroids = Square::with_capacity(d, c);
    for row in rand::seq::index::sample(&mut init_rng, n, c.min(n)) {
        centroids.push_slice(samples.row(row));
    }
    if n == 0 && c == 1 {
        centroids.push_iter(std::iter::repeat_n(0.0, d));
    }
    while centroids.len() < c {
        centroids.push_iter((0..d).map(|_| init_rng.random_range(-1.0f32..1.0f32)));
    }
    if n <= c {
        if is_spherical {
            pool.install(|| {
                centroids.par_iter_mut().for_each(|row| {
                    normalize(row);
                })
            });
        }
        return centroids;
    }

    let (batch_size, alpha0) = recipe(n, d, c, epochs);
    let total_steps = epochs
        .checked_mul(n.div_ceil(batch_size))
        .expect("too many e-means updates");
    let mut shuffle_rng = StdRng::from_rng(&mut init_rng);
    let mut revival_rng = StdRng::from_rng(&mut init_rng);
    let mut order: Vec<_> = (0..n).collect();
    let mut targets = vec![0; batch_size];
    let mut stats: Vec<_> = (0..c)
        .map(|_| ClusterStats {
            members: Vec::new(),
            usage: INITIAL_USAGE,
            closing_count: 0,
        })
        .collect();
    let mut projected = is_spherical.then(|| Square::from_zeros(d, c));
    let mut closing_sums = Square::from_zeros(d, c);
    let mut step = 0;

    for epoch in 0..epochs {
        order.shuffle(&mut shuffle_rng);
        let closing = epoch + 1 == epochs;
        for batch in order.chunks(batch_size) {
            on_batch(epoch);
            let mut has_zero = false;
            let assignment_centroids = match projected.as_mut() {
                Some(projected) => {
                    has_zero = pool.install(|| {
                        projected
                            .par_iter_mut()
                            .zip_eq(centroids.par_iter())
                            .map(|(unit, raw)| {
                                unit.copy_from_slice(raw);
                                !normalize(unit)
                            })
                            .reduce(|| false, |a, b| a || b)
                    });
                    &*projected
                }
                None => &centroids,
            };
            let targets = &mut targets[..batch.len()];
            {
                let index: Box<dyn Fn(&[f32]) -> (f32, usize) + Sync + '_> =
                    if use_rabitq && !has_zero {
                        Box::new(rabitq_index(pool, assignment_centroids))
                    } else if is_spherical {
                        Box::new(dot_index(assignment_centroids))
                    } else {
                        Box::new(flat_index(assignment_centroids))
                    };
                pool.install(|| {
                    targets
                        .par_iter_mut()
                        .zip_eq(batch.par_iter())
                        .for_each(|(target, &row)| {
                            *target = index(samples.row(row)).1;
                        });
                });
            }
            for (&row, &target) in batch.iter().zip(targets.iter()) {
                stats[target].members.push(row);
            }

            // Each centroid owns its rows, so sums can be reduced in parallel
            // without atomics or one full centroid matrix per worker.
            let alpha = scheduled_alpha(alpha0, step, total_steps);
            pool.install(|| {
                centroids
                    .par_iter_mut()
                    .zip_eq(stats.par_iter_mut())
                    .zip_eq(closing_sums.par_iter_mut())
                    .for_each_init(
                        || vec![0.0; d],
                        |sum, ((centroid, stats), closing_sum)| {
                            update_centroid(
                                centroid,
                                stats,
                                sum,
                                closing_sum,
                                &samples,
                                closing,
                                alpha,
                            );
                        },
                    );
            });
            step += 1;
            if d > 2 && step.is_multiple_of(REVIVAL_INTERVAL) {
                revive_unused(&mut centroids, &mut stats, &mut revival_rng);
            }
        }
    }

    pool.install(|| {
        centroids
            .par_iter_mut()
            .zip_eq(stats.par_iter())
            .zip_eq(closing_sums.par_iter())
            .for_each(|((centroid, stats), closing_sum)| {
                if stats.closing_count != 0 {
                    centroid.copy_from_slice(closing_sum);
                    f32::vector_mul_scalar_inplace(centroid, 1.0 / stats.closing_count as f32);
                }
                if use_rabitq {
                    rabitq::rotate::rotate_reversed_inplace(centroid);
                }
                if is_spherical {
                    normalize(centroid);
                }
            });
    });
    centroids
}

fn recipe(n: usize, d: usize, c: usize, epochs: usize) -> (usize, f64) {
    let quality = d <= 16;
    let coefficient = if quality { 0.00439175 } else { 0.035134 };
    let raw_batch = coefficient
        * (c as f64).powf(0.134749)
        * (n as f64).powf(0.876473)
        * (epochs as f64).powf(0.813344);
    let cap = if quality {
        n / 2
    } else {
        nearest_power_of_two(n as f64 / 2.0)
    };
    let batch_size = nearest_power_of_two(raw_batch).min(cap).clamp(1, n);
    let alpha0 = 0.8244
        * (c as f64).powf(-0.4617)
        * (batch_size as f64).powf(0.9003)
        * (n as f64).powf(-0.4498)
        * (epochs as f64).powf(-0.1057);
    (batch_size, alpha0.clamp(1e-4, 1.0))
}

fn nearest_power_of_two(value: f64) -> usize {
    let exponent = value.log2().round().max(0.0) as u32;
    1usize.checked_shl(exponent).unwrap_or(usize::MAX)
}

fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * step as f64 / total_steps as f64).cos());
    alpha0 * (0.1 + 0.9 * cosine)
}

fn normalize(row: &mut [f32]) -> bool {
    let norm = f32::reduce_sum_of_x2(row).sqrt();
    if norm.is_normal() {
        f32::vector_mul_scalar_inplace(row, 1.0 / norm);
        return true;
    }
    let max = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max == 0.0 {
        return false;
    }
    for x in row.iter_mut() {
        *x /= max;
    }
    let inverse_norm = 1.0 / f32::reduce_sum_of_x2(row).sqrt();
    f32::vector_mul_scalar_inplace(row, inverse_norm);
    true
}

fn dot_index(centroids: &Square) -> impl Fn(&[f32]) -> (f32, usize) + Sync {
    move |sample| {
        let mut result = (f32::INFINITY, 0);
        for (i, centroid) in centroids.into_iter().enumerate() {
            let distance = -f32::reduce_sum_of_xy(sample, centroid);
            if distance < result.0 {
                result = (distance, i);
            }
        }
        result
    }
}

fn update_centroid(
    centroid: &mut [f32],
    stats: &mut ClusterStats,
    sum: &mut [f32],
    closing_sum: &mut [f32],
    samples: &SquareMut<'_>,
    closing: bool,
    alpha: f64,
) {
    if alpha != 1.0 {
        stats.usage *= (1.0 - alpha) as f32;
    }
    let count = stats.members.len();
    if count == 0 {
        return;
    }
    sum.fill(0.0);
    for row in stats.members.drain(..) {
        f32::vector_add_inplace(sum, samples.row(row));
    }
    if closing {
        stats.closing_count += count;
        f32::vector_add_inplace(closing_sum, sum);
    }
    let incoming = alpha as f32 * count as f32;
    let retained = if alpha == 1.0 { 0.0 } else { stats.usage };
    stats.usage = retained + incoming;
    let weight = incoming / stats.usage;
    f32::vector_mul_scalar_inplace(sum, 1.0 / count as f32);
    if weight == 1.0 {
        centroid.copy_from_slice(sum);
    } else {
        for (x, &mean) in centroid.iter_mut().zip(sum.iter()) {
            *x += weight * (mean - *x);
        }
    }
}

fn revive_unused(centroids: &mut Square, stats: &mut [ClusterStats], rng: &mut StdRng) {
    let mean_usage = stats.iter().map(|stats| stats.usage).sum::<f32>() / stats.len() as f32;
    let threshold = UNUSED_THRESHOLD * mean_usage;
    let unused: Vec<_> = (0..stats.len())
        .filter(|&i| stats[i].usage < threshold)
        .collect();
    if unused.is_empty() {
        return;
    }
    // Gumbel top-k samples donors without replacement, weighted by usage.
    let mut donors: Vec<_> = stats
        .iter()
        .enumerate()
        .filter_map(|(i, stats)| {
            let usage = stats.usage;
            (usage >= threshold && usage > 0.0).then(|| {
                let uniform = rng.random_range(f64::MIN_POSITIVE..1.0);
                (i, f64::from(usage).ln() - (-uniform.ln()).ln())
            })
        })
        .collect();
    donors.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    for (target, (source, _)) in unused.into_iter().zip(donors) {
        centroids.copy_within(source..source + 1, target);
        for j in 0..centroids.d() {
            let delta = if j % 2 == 0 {
                SPLIT_EPSILON
            } else {
                -SPLIT_EPSILON
            };
            centroids[source][j] *= 1.0 - delta;
            centroids[target][j] *= 1.0 + delta;
        }
        if centroids[source] == centroids[target] {
            let value = centroids[target][0];
            centroids[target][0] = if value == f32::MAX {
                value.next_down()
            } else {
                value.next_up()
            };
        }
        let half = stats[source].usage * 0.5;
        stats[source].usage = half;
        stats[target].usage = half;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_mean_includes_every_row_in_the_final_pass() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let mut samples = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0];
        let mut batches = Vec::new();
        let centroids = e_means(
            &pool,
            2,
            SquareMut::new(2, &mut samples),
            1,
            [7; 32],
            false,
            2,
            |epoch| batches.push(epoch),
        );
        assert_eq!(batches, [0, 0, 0, 1, 1, 1]);
        assert_eq!(centroids.len(), 1);
        assert_eq!(&centroids[0], &[4.0, 5.0]);
    }

    #[test]
    fn empty_and_zero_spherical_samples_have_finite_centroids() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for mut samples in [vec![], vec![0.0; 18]] {
            let centroids = e_means(
                &pool,
                3,
                SquareMut::new(3, &mut samples),
                1,
                [7; 32],
                true,
                2,
                |_| {},
            );
            assert_eq!(centroids.len(), 1);
            assert_eq!(&centroids[0], &[0.0; 3]);
        }
    }

    #[test]
    fn recipe_matches_the_faiss_defaults() {
        let (batch_size, alpha0) = recipe(167_689, 4_096, 1_637, 5);
        assert_eq!(batch_size, 16_384);
        assert!((alpha0 - 0.634_732_435_7).abs() < 1e-9);
        assert_eq!(recipe(3, 2, 1, 2).0, 1);
    }
}
