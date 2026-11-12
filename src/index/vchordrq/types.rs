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

use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};
use vchordrq::types::VchordrqIndexOptions;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqDefaultBuildOptions {}

#[allow(clippy::derivable_impls)]
impl Default for VchordrqDefaultBuildOptions {
    fn default() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum KMeansAlgorithm {
    Lloyd {},
    Hierarchical {},
    #[serde(rename = "e_means")]
    EMeans(EMeansOptions),
}

impl Validate for KMeansAlgorithm {
    fn validate(&self) -> Result<(), ValidationErrors> {
        match self {
            Self::EMeans(options) => options.validate(),
            Self::Lloyd {} | Self::Hierarchical {} => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct EMeansOptions {
    #[serde(default = "EMeansOptions::default_epochs")]
    #[validate(range(min = 1, max = 1024))]
    pub epochs: u32,
}

impl EMeansOptions {
    fn default_epochs() -> u32 {
        5
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqInternalBuildOptions {
    #[serde(default = "VchordrqInternalBuildOptions::default_lists")]
    #[validate(length(min = 0, max = 8), custom(function = VchordrqInternalBuildOptions::validate_lists))]
    pub lists: Vec<u32>,
    #[serde(default = "VchordrqInternalBuildOptions::default_spherical_centroids")]
    pub spherical_centroids: bool,
    #[serde(default = "VchordrqInternalBuildOptions::default_sampling_factor")]
    #[validate(range(min = 1, max = 1024))]
    pub sampling_factor: u32,
    #[serde(default = "VchordrqInternalBuildOptions::default_kmeans_iterations")]
    #[validate(range(min = 0, max = 1024))]
    pub kmeans_iterations: u32,
    #[serde(default = "VchordrqInternalBuildOptions::default_build_threads")]
    #[validate(range(min = 1, max = 255))]
    pub build_threads: u16,
    #[serde(default = "VchordrqInternalBuildOptions::default_kmeans_algorithm")]
    #[validate(nested)]
    pub kmeans_algorithm: KMeansAlgorithm,
    #[serde(default = "VchordrqInternalBuildOptions::default_kmeans_dimension")]
    #[validate(range(min = 1, max = 16000))]
    pub kmeans_dimension: Option<u32>,
}

impl VchordrqInternalBuildOptions {
    fn default_lists() -> Vec<u32> {
        Vec::new()
    }
    fn validate_lists(lists: &[u32]) -> Result<(), ValidationError> {
        if !lists.is_sorted() {
            return Err(ValidationError::new("`lists` should be in ascending order"));
        }
        if !lists.iter().all(|x| (1..=1 << 24).contains(x)) {
            return Err(ValidationError::new("list is too long or too short"));
        }
        Ok(())
    }
    fn default_spherical_centroids() -> bool {
        false
    }
    fn default_sampling_factor() -> u32 {
        256
    }
    fn default_kmeans_iterations() -> u32 {
        10
    }
    fn default_build_threads() -> u16 {
        1
    }
    fn default_kmeans_algorithm() -> KMeansAlgorithm {
        KMeansAlgorithm::Lloyd {}
    }
    fn default_kmeans_dimension() -> Option<u32> {
        None
    }
}

impl Default for VchordrqInternalBuildOptions {
    fn default() -> Self {
        Self {
            lists: Self::default_lists(),
            spherical_centroids: Self::default_spherical_centroids(),
            sampling_factor: Self::default_sampling_factor(),
            kmeans_iterations: Self::default_kmeans_iterations(),
            build_threads: Self::default_build_threads(),
            kmeans_algorithm: Self::default_kmeans_algorithm(),
            kmeans_dimension: Self::default_kmeans_dimension(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqExternalBuildOptions {
    #[validate(custom(function = VchordrqExternalBuildOptions::validate_table))]
    pub table: String,
}

impl VchordrqExternalBuildOptions {
    fn validate_table(table: &str) -> Result<(), ValidationError> {
        let (schema_name, table_name) = if let Some((left, right)) = table.split_once(".") {
            (Some(left), right)
        } else {
            (None, table)
        };
        fn check(s: &str) -> bool {
            if s.is_empty() {
                return false;
            }
            if !matches!(s.as_bytes()[0],  b'A'..=b'Z' | b'a'..=b'z' | b'_') {
                return false;
            }
            for c in s.as_bytes().iter().copied() {
                if !matches!(c,  b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z' | b'_' | b'$') {
                    return false;
                }
            }
            true
        }
        if let Some(schema_name) = schema_name {
            if !check(schema_name) {
                return Err(ValidationError::new("table name is not well-formed"));
            }
        }
        if !check(table_name) {
            return Err(ValidationError::new("table name is not well-formed"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum VchordrqBuildSourceOptions {
    Default(VchordrqDefaultBuildOptions),
    Internal(VchordrqInternalBuildOptions),
    External(VchordrqExternalBuildOptions),
}

impl Default for VchordrqBuildSourceOptions {
    fn default() -> Self {
        Self::Default(Default::default())
    }
}

impl Validate for VchordrqBuildSourceOptions {
    fn validate(&self) -> Result<(), ValidationErrors> {
        use VchordrqBuildSourceOptions::*;
        match self {
            Default(default_build) => default_build.validate(),
            Internal(internal_build) => internal_build.validate(),
            External(external_build) => external_build.validate(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqBuildOptions {
    #[serde(flatten)]
    #[validate(nested)]
    pub source: VchordrqBuildSourceOptions,
    #[serde(deserialize_with = "VchordrqBuildOptions::deserialize_pin")]
    #[serde(default = "VchordrqBuildOptions::default_pin")]
    #[validate(range(min = -1, max = 2))]
    pub pin: i32,
}

impl VchordrqBuildOptions {
    pub fn deserialize_pin<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<i32, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Untagged {
            Bool(bool),
            I32(i32),
        }

        match Untagged::deserialize(deserializer)? {
            Untagged::Bool(b) => Ok(if b { 1 } else { -1 }),
            Untagged::I32(i) => Ok(i),
        }
    }
    pub fn default_pin() -> i32 {
        -1
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct VchordrqIndexingOptions {
    #[serde(flatten)]
    #[validate(nested)]
    pub index: VchordrqIndexOptions,
    #[serde(default)]
    #[validate(nested)]
    pub build: VchordrqBuildOptions,
}
