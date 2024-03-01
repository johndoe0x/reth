use derive_more::{Deref, DerefMut};
use rayon::iter::IntoParallelIterator;
use reth_primitives::B256;
use reth_trie::{prefix_set::PrefixSet, HashedPostState};
use std::collections::HashMap;

#[derive(Deref, DerefMut, Debug)]
pub(crate) struct StorageRootTargets(HashMap<B256, PrefixSet>);

impl StorageRootTargets {
    /// Create new storage root targets from updated post state accounts
    /// and storage prefix sets.
    ///
    /// NOTE: Since updated accounts and prefix sets always overlap,
    /// it's important that iterator over storage prefix sets takes precedence.
    pub(crate) fn new(
        hashed_state: &HashedPostState,
        storage_prefix_sets: HashMap<B256, PrefixSet>,
    ) -> Self {
        let account_targets =
            hashed_state.accounts.keys().map(|address| (*address, PrefixSet::default()));
        Self(account_targets.chain(storage_prefix_sets).collect())
    }
}

impl IntoIterator for StorageRootTargets {
    type Item = (B256, PrefixSet);
    type IntoIter = std::collections::hash_map::IntoIter<B256, PrefixSet>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl IntoParallelIterator for StorageRootTargets {
    type Item = (B256, PrefixSet);
    type Iter = rayon::collections::hash_map::IntoIter<B256, PrefixSet>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.into_par_iter()
    }
}
