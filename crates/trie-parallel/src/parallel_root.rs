use crate::StorageRootTargets;
use alloy_rlp::{BufMut, Encodable};
use rayon::prelude::*;
use reth_db::database::Database;
use reth_primitives::{
    trie::{HashBuilder, Nibbles, TrieAccount},
    B256,
};
use reth_provider::{
    providers::{ConsistentDbView, ConsistentViewError},
    DatabaseProviderFactory, ProviderError,
};
use reth_trie::{
    hashed_cursor::{HashedCursorFactory, HashedPostStateCursorFactory},
    node_iter::{AccountNode, AccountNodeIter},
    trie_cursor::TrieCursorFactory,
    updates::TrieUpdates,
    walker::TrieWalker,
    HashedPostState, StorageRoot, StorageRootError,
};
use std::collections::HashMap;
use thiserror::Error;
use tracing::*;

#[derive(Debug)]
pub struct ParallelStateRoot<DB, Provider> {
    /// Consistent view of the database.
    view: ConsistentDbView<DB, Provider>,
    /// Changed hashed state.
    hashed_state: HashedPostState,
}

impl<DB, Provider> ParallelStateRoot<DB, Provider> {
    /// Create new parallel state root calculator.
    pub fn new(view: ConsistentDbView<DB, Provider>, hashed_state: HashedPostState) -> Self {
        Self { view, hashed_state }
    }
}

impl<DB, Provider> ParallelStateRoot<DB, Provider>
where
    DB: Database,
    Provider: DatabaseProviderFactory<DB> + Send + Sync,
{
    /// Calculate incremental state root in parallel.
    pub fn incremental_root(self) -> Result<B256, ParallelStateRootError> {
        self.calculate(false).map(|(root, _)| root)
    }

    /// Calculate incremental state root with updates in parallel.
    pub fn incremental_root_with_updates(
        self,
    ) -> Result<(B256, TrieUpdates), ParallelStateRootError> {
        self.calculate(true)
    }

    fn calculate(
        self,
        retain_updates: bool,
    ) -> Result<(B256, TrieUpdates), ParallelStateRootError> {
        let prefix_sets = self.hashed_state.construct_prefix_sets();
        let storage_root_targets =
            StorageRootTargets::new(&self.hashed_state, prefix_sets.storage_prefix_sets);
        let hashed_state_sorted = self.hashed_state.into_sorted();

        // Pre-calculate storage roots in parallel for accounts which were changed.
        debug!(target: "trie::parallel_state_root", len = storage_root_targets.len(), "pre-calculating storage roots");
        let mut storage_roots = storage_root_targets
            .into_par_iter()
            .map(|(hashed_address, prefix_set)| {
                let provider_ro = self.view.provider_ro()?;
                let storage_root_result = StorageRoot::new_hashed(
                    provider_ro.tx_ref(),
                    HashedPostStateCursorFactory::new(provider_ro.tx_ref(), &hashed_state_sorted),
                    hashed_address,
                )
                .with_prefix_set(prefix_set)
                .calculate(retain_updates);
                Ok((hashed_address, storage_root_result?))
            })
            .collect::<Result<HashMap<_, _>, ParallelStateRootError>>()?;

        trace!(target: "trie::parallel_state_root", "calculating state root");
        let mut trie_updates = TrieUpdates::default();

        let provider_ro = self.view.provider_ro()?;
        let hashed_cursor_factory =
            HashedPostStateCursorFactory::new(provider_ro.tx_ref(), &hashed_state_sorted);
        let trie_cursor_factory = provider_ro.tx_ref();

        let hashed_account_cursor =
            hashed_cursor_factory.hashed_account_cursor().map_err(ProviderError::Database)?;
        let trie_cursor =
            trie_cursor_factory.account_trie_cursor().map_err(ProviderError::Database)?;

        let walker = TrieWalker::new(trie_cursor, prefix_sets.account_prefix_set)
            .with_updates(retain_updates);
        let mut account_node_iter = AccountNodeIter::new(walker, hashed_account_cursor);
        let mut hash_builder = HashBuilder::default().with_updates(retain_updates);

        let mut account_rlp = Vec::with_capacity(128);
        while let Some(node) = account_node_iter.try_next().map_err(ProviderError::Database)? {
            match node {
                AccountNode::Branch(node) => {
                    hash_builder.add_branch(node.key, node.value, node.children_are_in_trie);
                }
                AccountNode::Leaf(hashed_address, account) => {
                    let (storage_root, _, updates) = match storage_roots.remove(&hashed_address) {
                        Some(result) => result,
                        None => StorageRoot::new_hashed(
                            trie_cursor_factory,
                            hashed_cursor_factory.clone(),
                            hashed_address,
                        )
                        .calculate(retain_updates)?,
                    };

                    if retain_updates {
                        trie_updates.extend(updates.into_iter());
                    }

                    account_rlp.clear();
                    let account = TrieAccount::from((account, storage_root));
                    account.encode(&mut account_rlp as &mut dyn BufMut);
                    hash_builder.add_leaf(Nibbles::unpack(hashed_address), &account_rlp);
                }
            }
        }

        let root = hash_builder.root();

        trie_updates.finalize_state_updates(
            account_node_iter.walker,
            hash_builder,
            prefix_sets.destroyed_accounts,
        );

        Ok((root, trie_updates))
    }
}

/// Error during parallel state root calculation.
#[derive(Error, Debug)]
pub enum ParallelStateRootError {
    /// Consistency error on attempt to create new database provider.
    #[error(transparent)]
    ConsistentView(#[from] ConsistentViewError),
    /// Error while calculating storage root.
    #[error(transparent)]
    StorageRoot(#[from] StorageRootError),
    /// Provider error.
    #[error(transparent)]
    Provider(#[from] ProviderError),
}
