use alloy_rlp::{BufMut, Encodable};
use reth_db::database::Database;
use reth_primitives::{
    trie::{HashBuilder, Nibbles, TrieAccount},
    B256,
};
use reth_provider::{
    providers::{ConsistentDbView, ConsistentViewError},
    DatabaseProviderFactory, ProviderError,
};
use reth_tasks::TaskSpawner;
use reth_trie::{
    hashed_cursor::{HashedCursorFactory, HashedPostStateCursorFactory},
    node_iter::{AccountNode, AccountNodeIter},
    prefix_set::PrefixSet,
    trie_cursor::TrieCursorFactory,
    updates::{TrieKey, TrieUpdates},
    walker::TrieWalker,
    HashedPostState, StorageRoot, StorageRootError,
};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;
use tokio::sync::oneshot;
use tracing::*;

pub struct AsyncStateRoot<DB, Provider> {
    /// Consistent view of the database.
    view: ConsistentDbView<DB, Provider>,
    /// Task spawner.
    task_spawner: Arc<dyn TaskSpawner>,
    /// Changed hashed state.
    hashed_state: HashedPostState,
}

impl<DB, Provider> AsyncStateRoot<DB, Provider> {
    pub fn new(
        view: ConsistentDbView<DB, Provider>,
        task_spawner: Arc<dyn TaskSpawner>,
        hashed_state: HashedPostState,
    ) -> Self {
        Self { view, task_spawner, hashed_state }
    }
}

impl<DB, Provider> AsyncStateRoot<DB, Provider>
where
    DB: Database + Clone + 'static,
    Provider: DatabaseProviderFactory<DB> + Clone + Send + 'static,
{
    pub async fn incremental_root(self) -> Result<B256, AsyncStateRootError> {
        self.calculate(false).await.map(|(root, _)| root)
    }

    pub async fn incremental_root_with_updates(
        self,
    ) -> Result<(B256, TrieUpdates), AsyncStateRootError> {
        self.calculate(true).await
    }

    async fn calculate(
        self,
        retain_updates: bool,
    ) -> Result<(B256, TrieUpdates), AsyncStateRootError> {
        let prefix_sets = self.hashed_state.construct_prefix_sets();
        let storage_root_targets = self
            .hashed_state
            .accounts
            .keys()
            .map(|address| (*address, PrefixSet::default()))
            .chain(prefix_sets.storage_prefix_sets)
            .collect::<HashMap<_, _>>();
        let hashed_state_sorted = Arc::new(self.hashed_state.into_sorted());

        // Pre-calculate storage roots async for accounts which were changed.
        debug!(target: "trie::async_state_root", len = storage_root_targets.len(), "pre-calculating storage roots");

        let mut storage_roots = HashMap::with_capacity(storage_root_targets.len());
        for (hashed_address, prefix_set) in storage_root_targets {
            let (tx, rx) = oneshot::channel();
            let view = self.view.clone();
            let hashed_state_sorted = hashed_state_sorted.clone();
            self.task_spawner.spawn(Box::pin(async move {
                let result = view
                    .provider_ro()
                    .map_err(AsyncStateRootError::ConsistentView)
                    .and_then(|provider_ro| {
                        let tx = provider_ro.tx_ref();
                        let calculator = StorageRoot::new_hashed(
                            tx,
                            HashedPostStateCursorFactory::new(tx, &hashed_state_sorted),
                            hashed_address,
                        )
                        .with_prefix_set(prefix_set);

                        Ok(if retain_updates {
                            let (root, _, updates) = calculator.root_with_updates()?;
                            (root, Some(updates))
                        } else {
                            (calculator.root()?, None)
                        })
                    });

                let _ = tx.send(result);
            }));
            storage_roots.insert(hashed_address, rx);
        }

        trace!(target: "trie::async_state_root", "calculating state root");
        let mut trie_updates = TrieUpdates::default();

        let provider_ro = self.view.provider_ro()?;
        let tx = provider_ro.tx_ref();
        let hashed_cursor_factory = HashedPostStateCursorFactory::new(tx, &hashed_state_sorted);
        let trie_cursor_factory = tx;

        let hashed_account_cursor =
            hashed_cursor_factory.hashed_account_cursor().map_err(ProviderError::Database)?;
        let trie_cursor =
            trie_cursor_factory.account_trie_cursor().map_err(ProviderError::Database)?;

        let walker = TrieWalker::new(trie_cursor, prefix_sets.account_prefix_set);
        let mut hash_builder = HashBuilder::default();
        let mut account_node_iter = AccountNodeIter::new(walker, hashed_account_cursor);

        account_node_iter.walker.set_updates(retain_updates);
        hash_builder.set_updates(retain_updates);

        let mut account_rlp = Vec::with_capacity(128);
        while let Some(node) = account_node_iter.try_next().map_err(ProviderError::Database)? {
            match node {
                AccountNode::Branch(node) => {
                    hash_builder.add_branch(node.key, node.value, node.children_are_in_trie);
                }
                AccountNode::Leaf(hashed_address, account) => {
                    let (storage_root, updates) = match storage_roots.remove(&hashed_address) {
                        Some(rx) => rx
                            .await
                            .map_err(|_| AsyncStateRootError::StorageRootChannelClosed)??,
                        None => {
                            let calculator = StorageRoot::new_hashed(
                                trie_cursor_factory,
                                hashed_cursor_factory.clone(),
                                hashed_address,
                            );

                            if retain_updates {
                                let (root, _, updates) = calculator.root_with_updates()?;
                                (root, Some(updates))
                            } else {
                                (calculator.root()?, None)
                            }
                        }
                    };

                    if let Some(updates) = updates {
                        trie_updates.extend(updates.into_iter());
                    }

                    let account = TrieAccount::from((account, storage_root));

                    account_rlp.clear();
                    account.encode(&mut account_rlp as &mut dyn BufMut);

                    hash_builder.add_leaf(Nibbles::unpack(hashed_address), &account_rlp);
                }
            }
        }

        let root = hash_builder.root();

        let (_, walker_updates) = account_node_iter.walker.split();
        let (_, hash_builder_updates) = hash_builder.split();

        trie_updates.extend(walker_updates);
        trie_updates.extend_with_account_updates(hash_builder_updates);
        trie_updates.extend_with_deletes(
            prefix_sets.destroyed_accounts.into_iter().map(TrieKey::StorageTrie),
        );

        Ok((root, trie_updates))
    }
}

#[derive(Error, Debug)]
pub enum AsyncStateRootError {
    #[error("storage root channel closed")]
    StorageRootChannelClosed,
    #[error(transparent)]
    ConsistentView(#[from] ConsistentViewError),
    #[error(transparent)]
    StorageRoot(#[from] StorageRootError),
    #[error(transparent)]
    Provider(#[from] ProviderError),
}
