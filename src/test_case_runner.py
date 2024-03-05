from typing import Sequence

from test_case_builder import TC
from test_case_builder import (
    SignedBeaconBlock, Attestation, AttesterSlashing,
    get_forkchoice_store, do_spec_step, Store, get_head, get_filtered_block_tree, get_weight,
    get_current_epoch, get_active_validator_indices, get_proposer_head, get_current_slot, hash_tree_root
)
from abstract_spec import (FCState, get_head as abs_get_head, get_head_ as abs_get_head_, get_weight as abs_get_weight,
    get_filtered_block_tree as abs_get_filtered_block_tree,
    get_forkchoice_state, validate_tick, update_on_tick, validate_block, update_on_block,
    validate_attestation, update_on_attestation, make_tree,
    validate_attester_slashing, update_on_attester_slashing)


def compare_fc_state(fc_state: FCState, store: Store) -> None:
    assert fc_state.time == store.time
    assert fc_state.get_finalized_checkpoint() == store.finalized_checkpoint
    assert fc_state.proposer_boost_root == store.proposer_boost_root
    assert fc_state.equivocating_indices == store.equivocating_indices
    assert fc_state.blocks == set(store.blocks.values())
    for block_root in store.block_states.keys():
        assert fc_state.get_block_state(block_root) == store.block_states[block_root]
    for checkpoint in store.checkpoint_states:
        assert fc_state.get_checkpoint_state(checkpoint) == store.checkpoint_states[checkpoint]
    assert fc_state.latest_messages == store.latest_messages
    for block_root in store.unrealized_justifications.keys():
        if block_root != fc_state.anchor_root:
            assert fc_state.get_unrealized_justification(block_root) == store.unrealized_justifications[block_root]
    filtered_block_tree = get_filtered_block_tree(store)
    tree, filtered_blocks = abs_get_filtered_block_tree(fc_state)
    assert filtered_blocks == set(filtered_block_tree.keys())
    state = fc_state.get_checkpoint_state(fc_state.get_justified_checkpoint())
    unslashed_and_active_indices = [
        i for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    weight_map = {}
    for block_root in filtered_block_tree.keys():
        weight = get_weight(store, block_root)
        assert abs_get_weight(fc_state, state, unslashed_and_active_indices, block_root) == weight
        weight_map[block_root] = weight
    assert abs_get_head_(fc_state, tree, weight_map) == get_head(store)



def do_abs_spec_step(fc_state, evt):
    match evt:
        case int(time):
            validate_tick(fc_state, time)
            fc_state = update_on_tick(fc_state, time)
        case SignedBeaconBlock() as signed_block:
            validate_block(fc_state, signed_block)
            fc_state = update_on_block(fc_state, signed_block)
        case Attestation() as attestation:
            validate_attestation(fc_state, attestation, False)
            fc_state = update_on_attestation(fc_state, attestation, False)
        case AttesterSlashing() as attester_slashing:
            validate_attester_slashing(fc_state, attester_slashing)
            fc_state = update_on_attester_slashing(fc_state, attester_slashing)
        case _:
            assert False
    return fc_state

def mk_initial_states(anchor):
    anchor_state, anchor_block = anchor
    store = get_forkchoice_store(anchor_state, anchor_block)
    fc_state = get_forkchoice_state(anchor_state, anchor_block)
    return fc_state, store

def check_head_root(store):
    head_root = get_head(store)
    try:
        return get_proposer_head(store, head_root, get_current_slot(store))
    except Exception:
        return head_root



def run_tc(tc: TC, ignore_exceptions=False):
    fc_state, store = mk_initial_states(tc.anchor)
    #compare_fc_state(fc_state, store)
    check_head_root(store)
    for evt in tc.events:
        do_spec_step(store, evt, ignore_exceptions)
        check_head_root(store)
        #fc_state = do_abs_spec_step(fc_state, evt)
        #compare_fc_state(fc_state, store)


def enumerate_stores(tc: TC) -> Sequence[Store]:
    anchor_state, anchor_block = tc.anchor
    store = get_forkchoice_store(anchor_state, anchor_block)
    yield store
    for e in tc.events:
        do_spec_step(store, e)
        yield store

