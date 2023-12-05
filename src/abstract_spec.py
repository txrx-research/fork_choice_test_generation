import functools
from dataclasses import dataclass, field, replace
from typing import Any, FrozenSet, Mapping, Dict, Sequence, Optional, Set, List as PyList
from functools import cache

from frozendict import frozendict

from eth2spec.phase0 import minimal as phase0
from eth2spec.phase0.minimal import config
from eth2spec.phase0.minimal import GENESIS_SLOT, GENESIS_EPOCH, INTERVALS_PER_SLOT, SLOTS_PER_EPOCH
from eth2spec.phase0.minimal import Slot, Epoch, Root, uint64, ValidatorIndex, Gwei
from eth2spec.phase0.minimal import Checkpoint, BeaconState, SignedBeaconBlock, BeaconBlock, Attestation, AttesterSlashing
from eth2spec.phase0.minimal import Store, LatestMessage
from eth2spec.phase0.minimal import hash_tree_root
from eth2spec.phase0.minimal import (process_slots as bc_process_slots, state_transition as bc_state_transition,
                                     process_justification_and_finalization as bc_process_justification_and_finalization,
                                     get_current_epoch, compute_start_slot_at_epoch, compute_epoch_at_slot,
                                     get_active_validator_indices, get_total_active_balance,
                                     get_indexed_attestation, is_valid_indexed_attestation, is_slashable_attestation_data
                                     )


block_state_cache = {}
state_transition_cache = {}


def requires(precondition):
    return lambda f: f


def process_slots_pure(state: BeaconState, slot: Slot) -> BeaconState:
    state = state.copy()
    bc_process_slots(state, slot)
    return state


def state_transition_pure(state: BeaconState, signed_block: SignedBeaconBlock, validate_result: bool=True) -> BeaconState:
    state = state.copy()
    bc_state_transition(state, signed_block, validate_result)
    return state


def process_justification_and_finalization_pure(state: BeaconState) -> BeaconState:
    state = state.copy()
    bc_process_justification_and_finalization(state)
    return state


@dataclass(eq=True,frozen=True)
class FCState(object):
    anchor_root: Root
    anchor_state: BeaconState
    time: uint64
    genesis_time: uint64
    proposer_boost_root: Root
    equivocating_indices: FrozenSet[ValidatorIndex]
    blocks: FrozenSet[BeaconBlock]
    latest_messages: Mapping[ValidatorIndex, LatestMessage]

    def get_anchor_checkpoint(self) -> Checkpoint:
        return Checkpoint(epoch=get_current_epoch(self.get_block_state(self.anchor_root)), root=self.anchor_root)

    def find_block(self, block_root: Root) -> Optional[BeaconBlock]:
        for block in self.blocks:
            if hash_tree_root(block) == block_root:
                return block
        assert None

    def get_block(self, block_root: Root) -> BeaconBlock:
        block = self.find_block(block_root)
        assert block is not None
        return block

    def has_block(self, block_root: Root) -> bool:
        return self.find_block(block_root) is not None

    def get_block_state(self, block_root: Root) -> BeaconState:
        if block_root in block_state_cache:
            return block_state_cache[block_root]
        if block_root == self.anchor_root:
            res = self.anchor_state
        else:
            block = self.get_block(block_root)
            parent_state = self.get_block_state(block.parent_root)
            res = state_transition_pure(parent_state, SignedBeaconBlock(message=block), False)
        block_state_cache[block_root] = res
        return res

    def get_checkpoint_state(self, target: Checkpoint) -> BeaconState:
        base_state = self.get_block_state(target.root)
        if base_state.slot < compute_start_slot_at_epoch(target.epoch):
            process_slots_pure(base_state, compute_start_slot_at_epoch(target.epoch))
        return base_state

    def get_unrealized_justification(self, block_root: Root) -> Checkpoint:
        assert block_root != self.anchor_root
        return get_next_epoch_state(self, block_root).current_justified_checkpoint

    def get_justified_checkpoint(self) -> Checkpoint:
        def get_vs(block_root: Root) -> Checkpoint:
            if block_root == self.anchor_root:
                return self.get_anchor_checkpoint()
            else:
                checkpoint = get_actual_state(self, block_root).current_justified_checkpoint
                if checkpoint.epoch == 0:
                    return self.get_anchor_checkpoint()
                else:
                    return checkpoint
        return max((get_vs(hash_tree_root(block)) for block in self.blocks), key=lambda ch: ch.epoch)

    def get_finalized_checkpoint(self) -> Checkpoint:
        def get_fin_chkpt(block_root: Root) -> Checkpoint:
            if block_root == self.anchor_root:
                return self.get_anchor_checkpoint()
            else:
                checkpoint = get_actual_state(self, block_root).finalized_checkpoint
                if checkpoint.epoch == 0:
                    return self.get_anchor_checkpoint()
                else:
                    return checkpoint
        return max((get_fin_chkpt(hash_tree_root(block)) for block in self.blocks), key=lambda ch: ch.epoch)


def is_previous_epoch_justified(store: FCState) -> bool:
    current_slot = get_current_slot(store)
    current_epoch = compute_epoch_at_slot(current_slot)
    return store.get_justified_checkpoint().epoch + 1 == current_epoch


def get_forkchoice_state(anchor_state: BeaconState, anchor_block: BeaconBlock) -> FCState:
    assert isinstance(anchor_state, BeaconState)
    assert anchor_block.state_root == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    proposer_boost_root = Root()
    return FCState(
        anchor_root=anchor_root,
        anchor_state=anchor_state,
        time=uint64(anchor_state.genesis_time + config.SECONDS_PER_SLOT * anchor_state.slot),
        genesis_time=anchor_state.genesis_time,
        proposer_boost_root=proposer_boost_root,
        equivocating_indices=frozenset(),
        blocks=frozenset({anchor_block}),
        latest_messages=frozendict({}),
    )


def get_slots_since_genesis(store: FCState) -> int:
    return (store.time - store.genesis_time) // config.SECONDS_PER_SLOT


def get_current_slot(store: FCState) -> Slot:
    return Slot(GENESIS_SLOT + get_slots_since_genesis(store))


def compute_slots_since_epoch_start(slot: Slot) -> int:
    return slot - compute_start_slot_at_epoch(compute_epoch_at_slot(slot))


def get_ancestor(store: FCState, root: Root, slot: Slot) -> Root:
    block = store.get_block(root)
    if block.slot > slot:
        return get_ancestor(store, block.parent_root, slot)
    return root


def get_checkpoint_block(store: FCState, root: Root, epoch: Epoch) -> Root:
    """
    Compute the checkpoint block for epoch ``epoch`` in the chain of block ``root``
    """
    epoch_first_slot = compute_start_slot_at_epoch(epoch)
    return get_ancestor(store, root, epoch_first_slot)


def get_weight(store: FCState, state: BeaconState, unslashed_and_active_indices: Sequence[ValidatorIndex], root: Root) -> Gwei:
    attestation_score = Gwei(sum(
        state.validators[i].effective_balance for i in unslashed_and_active_indices
        if (i in store.latest_messages
            and i not in store.equivocating_indices
            and get_ancestor(store, store.latest_messages[i].root, store.get_block(root).slot) == root)
    ))
    if store.proposer_boost_root == Root():
        # Return only attestation score if ``proposer_boost_root`` is not set
        return attestation_score

    # Calculate proposer score if ``proposer_boost_root`` is set
    proposer_score = Gwei(0)
    # Boost is applied if ``root`` is an ancestor of ``proposer_boost_root``
    if get_ancestor(store, store.proposer_boost_root, store.get_block(root).slot) == root:
        committee_weight = get_total_active_balance(state) // SLOTS_PER_EPOCH
        proposer_score = (committee_weight * config.PROPOSER_SCORE_BOOST) // 100
    return attestation_score + proposer_score


def get_next_epoch_state(store: FCState, block_root: Root) -> BeaconState:
    state = store.get_block_state(block_root)
    return process_justification_and_finalization_pure(state)


def get_actual_state(store: FCState, block_root: Root) -> BeaconState:
    """
    Compute the voting source checkpoint in event that block with root ``block_root`` is the head block
    """
    # assert block_root != store.anchor_root
    block = store.get_block(block_root)
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    block_epoch = compute_epoch_at_slot(block.slot)
    if current_epoch > block_epoch:
        # The block is from a prior epoch, the voting source will be pulled-up
        return get_next_epoch_state(store, block_root)
    else:
        # The block is not from a prior epoch, therefore the voting source is not pulled up
        return store.get_block_state(block_root)


def get_voting_source(store: FCState, block_root: Root) -> Checkpoint:
    """
    Compute the voting source checkpoint in event that block with root ``block_root`` is the head block
    """
    return get_actual_state(store, block_root).current_justified_checkpoint


def make_tree(store: FCState, start: Root) -> Dict[Root, Sequence[Root]]:
    res = {}
    for b in store.blocks:
        root = hash_tree_root(b)
        parent_root = b.parent_root
        if parent_root not in res:
            res[parent_root] = []
        res[parent_root].append(root)

    res2 = {}
    def shake(root: Root):
        res2[root] = res.get(root, [])
        for ch in res2[root]:
            shake(ch)
    shake(start)
    return res2

def filter_block_tree(store: FCState, block_root: Root, tree: Dict[Root,Sequence[Root]], blocks: Set[Root]) -> bool:
    children = tree[block_root]

    # If any children branches contain expected finalized/justified checkpoints,
    # add to filtered block-tree and signal viability to parent.
    if any(children):
        filtered_children = [child for child in children if filter_block_tree(store, child, tree, blocks)]
        if any(filtered_children):
            blocks.add(block_root)
            return True
        else:
            return False

    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    voting_source = get_voting_source(store, block_root)

    # The voting source should be at the same height as the store's justified checkpoint
    correct_justified = (
            store.get_justified_checkpoint().epoch == GENESIS_EPOCH
            or voting_source.epoch == store.get_justified_checkpoint().epoch
    )

    # If the previous epoch is justified, the block should be pulled-up. In this case, check that unrealized
    # justification is higher than the store and that the voting source is not more than two epochs ago
    if not correct_justified and is_previous_epoch_justified(store):
        correct_justified = (
                store.get_unrealized_justification(block_root).epoch >= store.get_justified_checkpoint().epoch and
                voting_source.epoch + 2 >= current_epoch
        )

    finalized_checkpoint_block = get_checkpoint_block(
        store,
        block_root,
        store.get_finalized_checkpoint().epoch,
    )

    correct_finalized = (
            store.get_finalized_checkpoint().epoch == GENESIS_EPOCH
            or store.get_finalized_checkpoint().root == finalized_checkpoint_block
    )

    # If expected finalized/justified, add to viable block-tree and signal viability to parent.
    res = correct_justified and correct_finalized
    if res:
        blocks.add(block_root)
    return res


def get_filtered_block_tree(store: FCState) -> (Mapping[Root, Sequence[Root]], Set[Root]):
    """
    Retrieve a filtered block tree from ``store``, only returning branches
    whose leaf state's justified/finalized info agrees with that in ``store``.
    """
    base = store.get_justified_checkpoint().root
    tree = make_tree(store, base)
    blocks: Set[Root] = set()
    filter_block_tree(store, base, tree, blocks)
    return tree, blocks


def get_head(store: FCState) -> Root:
    # Get filtered block tree that only includes viable branches
    base = store.get_justified_checkpoint().root
    tree = make_tree(store, base)
    blocks = get_filtered_block_tree(store)
    state = store.get_checkpoint_state(store.get_justified_checkpoint())
    unslashed_and_active_indices = [
        i for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    weight_map = {
        block_root: get_weight(store, state, unslashed_and_active_indices, block_root)
        for block_root in blocks
    }
    return get_head_(store, tree, weight_map)


def get_head_(store: FCState, tree: Dict[Root, Sequence[Root]], weight_map: Dict[Root, Gwei]) -> Root:
    # Execute the LMD-GHOST fork choice
    head = store.get_justified_checkpoint().root
    while True:
        children = [child for child in tree[head] if child in weight_map]
        if len(children) == 0:
            return head
        # Sort by latest attesting balance with ties broken lexicographically
        # Ties broken by favoring block with lexicographically higher root
        head = max(children, key=lambda root: (weight_map[root], root))


def on_tick_per_slot(store: FCState, time: uint64) -> FCState:
    previous_slot = get_current_slot(store)

    # Update store time
    store = replace(store, time=time)

    current_slot = get_current_slot(store)

    # If this is a new slot, reset store.proposer_boost_root
    if current_slot > previous_slot:
        store = replace(store, proposer_boost_root=Root())

    return store


def validate_target_epoch_against_current_time(store: FCState, attestation: Attestation) -> None:
    target = attestation.data.target

    # Attestations must be from the current or previous epoch
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    # Use GENESIS_EPOCH for previous when genesis to avoid underflow
    previous_epoch = current_epoch - 1 if current_epoch > GENESIS_EPOCH else GENESIS_EPOCH
    # If attestation target is from a future epoch, delay consideration until the epoch arrives
    assert target.epoch in [current_epoch, previous_epoch]


def validate_on_attestation(store: FCState, attestation: Attestation, is_from_block: bool) -> None:
    target = attestation.data.target

    # If the given attestation is not from a beacon block message, we have to check the target epoch scope.
    if not is_from_block:
        validate_target_epoch_against_current_time(store, attestation)

    # Check that the epoch number and slot number are matching
    assert target.epoch == compute_epoch_at_slot(attestation.data.slot)

    # Attestation target must be for a known block. If target block is unknown, delay consideration until block is found
    assert store.has_block(target.root)

    # Attestations must be for a known block. If block is unknown, delay consideration until the block is found
    assert store.has_block(attestation.data.beacon_block_root)
    # Attestations must not be for blocks in the future. If not, the attestation should not be considered
    assert store.get_block(attestation.data.beacon_block_root).slot <= attestation.data.slot

    # LMD vote must be consistent with FFG vote target
    assert target.root == get_checkpoint_block(store, attestation.data.beacon_block_root, target.epoch)

    # Attestations can only affect the fork choice of subsequent slots.
    # Delay consideration in the fork choice until their slot is in the past.
    assert get_current_slot(store) >= attestation.data.slot + 1


def update_latest_messages(store: FCState, attesting_indices: Sequence[ValidatorIndex], attestation: Attestation) -> FCState:
    target = attestation.data.target
    beacon_block_root = attestation.data.beacon_block_root
    non_equivocating_attesting_indices = [i for i in attesting_indices if i not in store.equivocating_indices]
    for i in non_equivocating_attesting_indices:
        if i not in store.latest_messages or target.epoch > store.latest_messages[i].epoch:
            store = replace(store, latest_messages=(store.latest_messages | { i: LatestMessage(epoch=target.epoch, root=beacon_block_root) }))
    return store


def validate_tick(store: FCState, time: uint64) -> None:
    assert time > store.time


# @requires(validate_tick)
def update_on_tick(store: FCState, time: uint64) -> FCState:
    validate_tick(store, time)
    # If the ``store.time`` falls behind, while loop catches up slot by slot
    # to ensure that every previous slot is processed with ``on_tick_per_slot``
    tick_slot = (time - store.genesis_time) // config.SECONDS_PER_SLOT
    while get_current_slot(store) < tick_slot:
        previous_time = store.genesis_time + (get_current_slot(store) + 1) * config.SECONDS_PER_SLOT
        store = on_tick_per_slot(store, previous_time)
    return on_tick_per_slot(store, time)


def validate_block(store: FCState, signed_block: SignedBeaconBlock) -> None:
    block = signed_block.message
    # Parent block must be known
    # assert store.has_block(block.parent_root)
    pre_state = store.get_block_state(block.parent_root)
    # Blocks cannot be in the future. If they are, their consideration must be delayed until they are in the past.
    assert get_current_slot(store) >= block.slot

    finalized_checkpoint = store.get_finalized_checkpoint()
    # Check block is a descendant of the finalized block at the checkpoint finalized slot
    finalized_checkpoint_block = get_checkpoint_block(
        store,
        block.parent_root,
        finalized_checkpoint.epoch,
    )
    assert finalized_checkpoint.root == finalized_checkpoint_block

    # Check the block is valid and compute the post-state
    state_transition_pure(pre_state, signed_block, True)


# @requires(validate_block)
def update_on_block(store: FCState, signed_block: SignedBeaconBlock) -> FCState:
    block = signed_block.message

    # Add new block to the store
    store = replace(store, blocks=(store.blocks | {block}))

    # Add proposer score boost if the block is timely
    time_into_slot = (store.time - store.genesis_time) % config.SECONDS_PER_SLOT
    is_before_attesting_interval = time_into_slot < config.SECONDS_PER_SLOT // INTERVALS_PER_SLOT
    if get_current_slot(store) == block.slot and is_before_attesting_interval:
        store = replace(store, proposer_boost_root=hash_tree_root(block))

    return store


def validate_attestation(store: FCState, attestation: Attestation, is_from_block: bool = False) -> None:
    validate_on_attestation(store, attestation, is_from_block)

    # Get state at the `target` to fully validate attestation
    target_state = store.get_checkpoint_state(attestation.data.target)
    indexed_attestation = get_indexed_attestation(target_state, attestation)
    assert is_valid_indexed_attestation(target_state, indexed_attestation)


# @requires(validate_attestation)
def update_on_attestation(store: FCState, attestation: Attestation, is_from_block: bool = False) -> FCState:
    """
    Run ``on_attestation`` upon receiving a new ``attestation`` from either within a block or directly on the wire.

    An ``attestation`` that is asserted as invalid may be valid at a later time,
    consider scheduling it for later processing in such case.
    """
    # Get state at the `target` to fully validate attestation
    target_state = store.get_checkpoint_state(attestation.data.target)
    indexed_attestation = get_indexed_attestation(target_state, attestation)

    # Update latest messages for attesting indices
    return update_latest_messages(store, indexed_attestation.attesting_indices, attestation)


def validate_attester_slashing(store: FCState, attester_slashing: AttesterSlashing) -> None:
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2
    assert is_slashable_attestation_data(attestation_1.data, attestation_2.data)
    state = store.get_block_state(store.get_justified_checkpoint().root)
    assert is_valid_indexed_attestation(state, attestation_1)
    assert is_valid_indexed_attestation(state, attestation_2)


# @requires(validate_attester_slashing)
def update_on_attester_slashing(store: FCState, attester_slashing: AttesterSlashing) -> FCState:
    """
    Run ``on_attester_slashing`` immediately upon receiving a new ``AttesterSlashing``
    from either within a block or directly on the wire.
    """
    attestation_1 = attester_slashing.attestation_1
    attestation_2 = attester_slashing.attestation_2

    indices = set(attestation_1.attesting_indices).intersection(attestation_2.attesting_indices)
    store = replace(store, equivocating_indices=store.equivocating_indices|indices)

    return store


def fc_state_transition(fc_state: FCState, e: (uint64|SignedBeaconBlock|Attestation|AttesterSlashing)) -> FCState:
    if isinstance(e, uint64):
        validate_tick(fc_state, e)
        return update_on_tick(fc_state, e)
    elif isinstance(e, SignedBeaconBlock):
        validate_block(fc_state, e)
        return update_on_block(fc_state, e)
        # process attestations from block
    elif isinstance(e, Attestation):
        validate_attestation(fc_state, e)
        return update_on_attestation(fc_state, e)
    elif isinstance(e, AttesterSlashing):
        validate_attester_slashing(fc_state, e)
        return update_on_attester_slashing(fc_state, e)
    else:
        assert False, "shouldn't happen"

