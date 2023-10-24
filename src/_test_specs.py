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

eth2spec.utils.bls.bls_active = False

def compare_fc_state(fc_state: FCState, store: Store) -> None:
    assert fc_state.time == store.time
    assert fc_state.genesis_time == store.genesis_time
    assert fc_state.get_justified_checkpoint() == store.justified_checkpoint, f"{fc_state.get_justified_checkpoint()} {store.justified_checkpoint}"
    assert fc_state.get_finalized_checkpoint() == store.finalized_checkpoint
    assert fc_state.proposer_boost_root == store.proposer_boost_root
    assert fc_state.equivocating_indices == store.equivocating_indices
    assert fc_state.blocks == set(store.blocks.values())
    for block_root in store.block_states.keys():
        assert fc_state.get_block_state(block_root) == store.block_states[block_root].freeze()
    for checkpoint in store.checkpoint_states:
        assert fc_state.get_checkpoint_state(checkpoint) == store.checkpoint_states[checkpoint].freeze()
    assert fc_state.latest_messages == store.latest_messages
    for block_root in store.unrealized_justifications.keys():
        if block_root != fc_state.anchor_root:
            assert fc_state.get_unrealized_justification(block_root) == store.unrealized_justifications[block_root]
    assert abs_get_head(fc_state) == get_head(store)
    assert abs_get_filtered_block_tree(fc_state) == get_filtered_block_tree(store)
    for block_root in get_filtered_block_tree(store).keys():
        assert abs_get_weight(fc_state, block_root) == get_weight(store, block_root)



def do_spec_step(store, evt):
    match evt:
        case int(time):
            on_tick(store, time)
        case Attestation() as attestation:
            on_attestation(store, attestation, False)
        case SignedBeaconBlock() as signed_block:
            on_block(store, signed_block)
        case AttesterSlashing() as attester_slashing:
            on_attester_slashing(store, attester_slashing)
        case _:
            assert False


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


def mk_anchor():
    no_validators = 16
    anchor_state = genesis.create_genesis_state(phase0, [MAX_EFFECTIVE_BALANCE] * no_validators, 0)
    anchor_block = BeaconBlock(state_root=hash_tree_root(anchor_state))
    return anchor_state, anchor_block


@dataclass(eq=True)
class TC:
    anchor: Tuple[BeaconState, BeaconBlock]
    events: Sequence[int|SignedBeaconBlock|Attestation|AttesterSlashing]


class TCBuilder:
    def __init__(self):
        self.anchor = mk_anchor()
        self.events = []
        self.store = get_forkchoice_store(*self.anchor)
        self.attestations = []

    def get_tc(self):
        return TC(self.anchor, self.events)

    def new_tick(self):
        slot = get_current_slot(self.store)
        tick = self.store.genesis_time + config.SECONDS_PER_SLOT*(slot - GENESIS_SLOT + 1)
        do_spec_step(self.store, tick)
        self.events.append(tick)

    def _get_last_block(self):
        return max(self.store.blocks.values(), key=lambda b: b.slot)

    def _resolve_block_ref(self, ref):
        if ref is None:
            return get_head(self.store)
        elif isinstance(ref, BeaconBlock):
            return hash_tree_root(ref)
        elif isinstance(ref, SignedBeaconBlock):
            return self._resolve_block_ref(ref.message)
        elif isinstance(ref, int):
            for root, block in self.store.blocks.items():
                if block.slot == ref:
                    return root
            assert False
        else:
            assert False

    def send_block(self, signed_block):
        do_spec_step(self.store, signed_block)
        self.events.append(signed_block)

    def mk_block(self, parent=None, atts=None):
        parent_root = self._resolve_block_ref(parent)
        slot = get_current_slot(self.store)
        state = self.store.block_states[parent_root].copy()
        block = block_helpers.build_empty_block(phase0, state, slot)
        if atts is not None:
            block.body.attestations.append(*atts)
        else:
            st = state.copy()
            phase0.process_slots(st, slot)
            selected_atts = []
            for a in self.attestations:
                assert isinstance(a, phase0.Attestation)
                if (a.data.source == st.current_justified_checkpoint
                            or a.data.source == st.previous_justified_checkpoint) \
                        and (a.data.slot + phase0.MIN_ATTESTATION_INCLUSION_DELAY <= slot <= a.data.slot + phase0.SLOTS_PER_EPOCH):
                    selected_atts.append(a)
            for a in selected_atts:
                block.body.attestations.append(a)


        block = block_helpers.transition_unsigned_block(phase0, state, block)
        block.state_root = hash_tree_root(state)
        signed_block = block_helpers.sign_block(phase0, state, block)
        return signed_block

    def new_block(self, parent=None, atts=None):
        self.send_block(self.mk_block(parent, atts))

    # def mk_attestation(self, vals):
    #     head = self._resolve_block_ref(None)
    #     slot = get_current_slot(self.store)
    #     epoch = compute_epoch_at_slot(slot)
    #     source = self.store.justified_checkpoint
    #     boundary_block = get_checkpoint_block(self.store, head, epoch)
    #     data = AttestationData(
    #         slot = slot, beacon_block_root=head, source=source,
    #         target=Checkpoint(epoch=epoch, root=boundary_block))
    #     return Attestation(data=data)

    def mk_attestations(self):
        head_root = self._resolve_block_ref(None)
        slot = get_current_slot(self.store)
        state = self.store.block_states[head_root].copy()
        if state.slot < slot:
            phase0.process_slots(state, slot)

        block_root = head_root
        current_epoch_start_slot = phase0.compute_start_slot_at_epoch(phase0.get_current_epoch(state))
        if slot < current_epoch_start_slot:
            epoch_boundary_root = phase0.get_block_root(state, phase0.get_previous_epoch(state))
        elif slot == current_epoch_start_slot:
            epoch_boundary_root = block_root
        else:
            epoch_boundary_root = phase0.get_block_root(state, phase0.get_current_epoch(state))

        if slot < current_epoch_start_slot:
            source_epoch = state.previous_justified_checkpoint.epoch
            source_root = state.previous_justified_checkpoint.root
        else:
            source_epoch = state.current_justified_checkpoint.epoch
            source_root = state.current_justified_checkpoint.root

        attestation_data = phase0.AttestationData(
            slot=slot,
            index=0,
            beacon_block_root=block_root,
            source=phase0.Checkpoint(epoch=source_epoch, root=source_root),
            target=phase0.Checkpoint(epoch=phase0.compute_epoch_at_slot(slot), root=epoch_boundary_root),
        )


        beacon_committee = phase0.get_beacon_committee(
            state,
            attestation_data.slot,
            attestation_data.index,
        )

        committee_size = len(beacon_committee)
        aggregation_bits = phase0.Bitlist[phase0.MAX_VALIDATORS_PER_COMMITTEE](*([0] * committee_size))
        attestation = phase0.Attestation(
            aggregation_bits=aggregation_bits,
            data=attestation_data,
        )
        attestations_helper.fill_aggregate_attestation(phase0, state, attestation, signed=False, filter_participant_set=None)

        atts = attestation
        return atts

    def send_attestation(self, att):
        do_spec_step(self.store, att)
        self.attestations.append(att)
        self.events.append(att)


def mk_initial_states(anchor):
    anchor_state, anchor_block = anchor
    store = get_forkchoice_store(anchor_state, anchor_block)
    fc_state = get_forkchoice_state(anchor_state, anchor_block)
    return fc_state, store


def run_tc(tc: TC):
    fc_state, store = mk_initial_states(tc.anchor)
    compare_fc_state(fc_state, store)
    for evt in tc.events:
        do_spec_step(store, evt)
        fc_state = do_abs_spec_step(fc_state, evt)
        compare_fc_state(fc_state, store)


def enumerate_stores(tc: TC) -> Sequence[Store]:
    anchor_state, anchor_block = tc.anchor
    store = get_forkchoice_store(anchor_state, anchor_block)
    yield store
    for e in tc.events:
        do_spec_step(store, e)
        yield store


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
    builder.new_tick()
    return builder.get_tc()


def _test_on_attestation():
    builder = TCBuilder()
    atts = builder.mk_attestations()
    builder.new_tick()
    builder.send_attestation(atts)
    return builder.get_tc()


def _test_on_attester_slashing():
    data_1 = AttestationData(0, Checkpoint(0, 0), Checkpoint(2, 0), 0)
    data_2 = AttestationData(0, Checkpoint(1, 0), Checkpoint(2, 0), 0)
    attester_slashing = AttesterSlashing(IndexedAttestation(data_1, (1,2)), IndexedAttestation(data_2, (2,3)))
    return TC(mk_anchor(), [attester_slashing])


def _test_on_block():
    builder = TCBuilder()
    for i in range(31):
        atts = builder.mk_attestations()
        builder.new_tick()
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


from minzinc_gen import generate_sm_link_test_cases

cases = [{'sources':[0,0,0,0],'targets':[1,2,3,4]}]
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

builder = TCBuilder()
for i in range(15):
    atts = builder.mk_attestations()
    builder.new_tick()
    builder.send_attestation(atts)

atts_prev = builder.mk_attestations()
builder.new_tick()
# 0
builder.send_attestation(atts_prev)
atts0 = builder.mk_attestations()
builder.new_tick()
# 1
builder.send_attestation(atts0)
atts1 = builder.mk_attestations()
builder.new_tick()
# 2
builder.send_attestation(atts1)
builder.new_block()
atts2 = builder.mk_attestations()
builder.new_tick()
# 3
builder.send_attestation(atts2)
builder.new_block()
atts3 = builder.mk_attestations()
builder.new_tick()
# 4
builder.send_attestation(atts3)
builder.new_block()
atts4 = builder.mk_attestations()
builder.new_tick()
# 5
builder.send_attestation(atts4)
builder.new_block()
atts5 = builder.mk_attestations()
builder.new_tick()
# 6
builder.send_attestation(atts5)
builder.new_block()
atts6 = builder.mk_attestations()
builder.new_tick()
# 7
builder.send_attestation(atts6)
builder.new_block()
atts7 = builder.mk_attestations()
builder.new_tick()
# 8


print()


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



