from dataclasses import dataclass, replace
from typing import Any, Sequence, Set, Tuple, Dict, Optional

import eth2spec.utils.bls
from frozendict import frozendict

from eth2spec.test.helpers import genesis, block as block_helpers, attestations as attestations_helper

from eth2spec.phase0 import minimal as phase0
from eth2spec.phase0.minimal import config, MAX_EFFECTIVE_BALANCE, GENESIS_SLOT, GENESIS_EPOCH
from eth2spec.phase0.minimal import (Store, LatestMessage, Attestation, AttesterSlashing, BeaconState,
                                     SignedBeaconBlock, BeaconBlock, Validator, Checkpoint, BeaconBlockBody,
                                     AttestationData)
from eth2spec.phase0.minimal import (get_head, get_filtered_block_tree, get_weight, on_tick, on_block, on_attestation,
                                     on_attester_slashing, get_forkchoice_store, get_current_slot, state_transition,
                                     process_slots, get_beacon_proposer_index, compute_epoch_at_slot,
                                     get_checkpoint_block, get_voting_source, is_previous_epoch_justified
                                     )
from eth2spec.phase0.minimal import hash_tree_root

from abstract_spec import FCState
from abstract_spec import (get_forkchoice_state, validate_tick, update_on_tick,
                           validate_block, update_on_block, validate_attestation, update_on_attestation,
                           validate_attester_slashing, update_on_attester_slashing,
                           get_head as abs_get_head, get_filtered_block_tree as abs_get_filtered_block_tree,
                           get_weight as abs_get_weight
                           )

from test_case_builder import TC, TCBuilder, do_spec_step

eth2spec.utils.bls.bls_active = False



def calc_block_predicate_coverage(store: Store, block_root):
    def get_children(root):
        for r, b in store.blocks.items():
            if b.parent_root == root:
                yield r

    def get_descendants(root):
        yield root
        for child in get_children(root):
            yield from get_descendants(child)

    leaf_block = not any(get_children(block_root))
    ancestor_of_justified_root = block_root in get_descendants(store.justified_checkpoint.root)
    sje_eq_genesis = store.justified_checkpoint.epoch == GENESIS_EPOCH
    voting_source = get_voting_source(store, block_root)
    vse_eq_sje = voting_source.epoch == store.justified_checkpoint.epoch
    prev_epoch_justified = is_previous_epoch_justified(store)
    buje_ge_sje = store.unrealized_justifications[block_root].epoch >= store.justified_checkpoint.epoch
    current_epoch = compute_epoch_at_slot(get_current_slot(store))
    vse_plus_2_ge_ce = voting_source.epoch + 2 >= current_epoch
    sfe_eq_genesis =store.finalized_checkpoint.epoch == GENESIS_EPOCH
    finalized_checkpoint_block = get_checkpoint_block(
        store,
        block_root,
        store.finalized_checkpoint.epoch,
    )
    ancestor_of_finalized_checkpoint = store.finalized_checkpoint.root == finalized_checkpoint_block

    return frozendict({
            'leaf_block': leaf_block,
            'ancestor_of_justified_root': ancestor_of_justified_root,
            'sje_eq_genesis': sje_eq_genesis,
            'vse_eq_sje': vse_eq_sje,
            'prev_epoch_justified': prev_epoch_justified,
            'buje_ge_sje': buje_ge_sje,
            'vse_plus_2_ge_ce': vse_plus_2_ge_ce,
            'sfe_eq_genesis': sfe_eq_genesis,
            'ancestor_of_finalized_checkpoint': ancestor_of_finalized_checkpoint
        })

def _test_anchor_state():
    return TCBuilder().get_tc()


def _test_simple_tick():
    builder = TCBuilder()
    builder.new_slot()
    return builder.get_tc()


def _test_on_attestation():
    builder = TCBuilder()
    atts = builder.mk_attestations()
    builder.new_slot()
    builder.send_attestation(atts)
    return builder.get_tc()


def _test_on_attester_slashing():
    builder = TCBuilder()
    data_1 = AttestationData(0, Checkpoint(0, 0), Checkpoint(2, 0), 0)
    data_2 = AttestationData(0, Checkpoint(1, 0), Checkpoint(2, 0), 0)
    attester_slashing = AttesterSlashing(IndexedAttestation(data_1, (1,2)), IndexedAttestation(data_2, (2,3)))
    return TC(builder.anchor, [attester_slashing])


def _test_on_block():
    builder = TCBuilder()
    for i in range(31):
        atts = builder.mk_attestations()
        builder.new_slot()
        builder.send_attestation(atts)
        builder.new_block(atts=(atts,))
    return builder.get_tc()


tcs = []
# tcs.append(b.get_tc())
# tcs.append(_test_anchor_state())
# tcs.append(_test_simple_tick())
# tcs.append(_test_on_attestation())
# tcs.append(_test_on_attester_slashing())
#tcs.append(_test_on_block())


# for case in cases: #generate_sm_link_test_cases():
#     sources = case['sources']
#     targets = case['targets']
#     sm_links = sorted(zip(sources, targets), key=lambda x: x[1])
#     parents = {}
#     builder = TCBuilder()
#     for src, tgt in sm_links:
#         parents[tgt*phase0.SLOTS_PER_EPOCH] = src*phase0.SLOTS_PER_EPOCH
#         while compute_epoch_at_slot(get_current_slot(builder.store)) < tgt:
#             atts = builder.mk_attestations()
#             builder.new_tick()
#             builder.send_attestation(atts)
#             builder.new_block(atts=(atts,))
#         atts = builder.mk_attestations()
#         builder.new_tick()
#         builder.send_attestation(atts)
#         builder.new_block(parent=src*phase0.SLOTS_PER_EPOCH, atts=None)
#     tcs.append(builder.get_tc())

# block_pred_cov = set()
# for tc in tcs:
#     for store in enumerate_stores(tc):
#         get_head(store)
#         for block_root in store.blocks.keys():
#             predicate_coverage = calc_block_predicate_coverage(store, block_root)
#             block_pred_cov.add(predicate_coverage)
#
# for c in block_pred_cov:
#     print(c)



