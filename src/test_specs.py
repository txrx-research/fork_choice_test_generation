from dataclasses import dataclass, replace
from typing import Any, Sequence, Set, Tuple, Dict, Optional
from frozendict import frozendict

from beacon_chain import GENESIS_SLOT, GENESIS_EPOCH, SECONDS_PER_SLOT, MAX_EFFECTIVE_BALANCE, SLOTS_PER_HISTORICAL_ROOT
from beacon_chain import Checkpoint, Validator, BeaconState, SignedBeaconBlock, BeaconBlock, Attestation, \
    AttesterSlashing, IndexedAttestation, AttestationData, ValidatorIndex, BeaconBlockBody
from beacon_chain import state_transition, compute_epoch_at_slot, get_current_epoch

from orig_spec import Store
from orig_spec import get_forkchoice_store, on_tick, on_block, on_attestation, on_attester_slashing,\
    get_current_slot, get_head, get_checkpoint_block, get_filtered_block_tree, get_weight, get_voting_source, \
    is_previous_epoch_justified

from utils import hash_tree_root

from abstract_spec import FCState
from abstract_spec import (get_forkchoice_state, validate_tick, update_on_tick,
                           validate_block, update_on_block, validate_attestation, update_on_attestation,
                           validate_attester_slashing, update_on_attester_slashing,
                           get_head as abs_get_head, get_filtered_block_tree as abs_get_filtered_block_tree,
                           get_weight as abs_get_weight
                           )


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



def do_spec_step(store, evt, tracer=None):
    match evt:
        case int(time):
            if tracer is not None:
                sys.settrace(tracer)
            try:
                on_tick(store, time)
            finally:
                if tracer is not None:
                    sys.settrace(None)
        case Attestation() as attestation:
            if tracer is not None:
                sys.settrace(tracer)
            try:
                on_attestation(store, attestation, False)
            finally:
                if tracer is not None:
                    sys.settrace(None)
        case SignedBeaconBlock() as signed_block:
            if tracer is not None:
                sys.settrace(tracer)
            try:
                on_block(store, signed_block)
            finally:
                if tracer is not None:
                    sys.settrace(None)
        case AttesterSlashing() as attester_slashing:
            if tracer is not None:
                sys.settrace(tracer)
            try:
                on_attester_slashing(store, attester_slashing)
            finally:
                if tracer is not None:
                    sys.settrace(None)
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
    validators = (Validator(False,MAX_EFFECTIVE_BALANCE),) * 16
    anchor_state = BeaconState(
        genesis_time=0,
        slot=0,
        validators=validators,
        current_epoch_attestations=[],
        previous_epoch_attestations=[],
        justification_bits=[0,0,0,0],
        block_roots=[0] * SLOTS_PER_HISTORICAL_ROOT,
        previous_justified_checkpoint=Checkpoint(-1, 0),
        current_justified_checkpoint=Checkpoint(-1, 0),
        finalized_checkpoint=Checkpoint(-1, 0))
    anchor_block = BeaconBlock(0, hash_tree_root(anchor_state), anchor_state.slot, BeaconBlockBody(()))
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

    def get_tc(self):
        return TC(self.anchor, self.events)

    def new_tick(self):
        slot = get_current_slot(self.store)
        tick = self.store.genesis_time + SECONDS_PER_SLOT*(slot - GENESIS_SLOT + 1)
        do_spec_step(self.store, tick)
        self.events.append(tick)

    def _get_last_block(self):
        return max(self.store.blocks.values(), key=lambda b: b.slot)

    def _resolve_block_ref(self, ref):
        if ref is None:
            #return hash_tree_root(self._get_last_block())
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

    def new_block(self, parent=None, atts=None):
        parent_root = self._resolve_block_ref(parent)
        slot = get_current_slot(self.store)
        block = BeaconBlock(parent_root, None, slot,BeaconBlockBody(atts or ()))
        pre_state = self.store.block_states[parent_root].copy()
        state_transition(pre_state, SignedBeaconBlock(block), False)
        state_root = hash_tree_root(pre_state)
        res_block = replace(block, state_root=state_root)
        signed_block = SignedBeaconBlock(res_block)
        self.send_block(signed_block)

    def mk_attestation(self, vals):
        head = self._resolve_block_ref(None)
        slot = get_current_slot(self.store)
        epoch = compute_epoch_at_slot(slot)
        source = self.store.justified_checkpoint
        boundary_block = get_checkpoint_block(self.store, head, epoch)
        data = AttestationData(slot, source, Checkpoint(epoch, boundary_block), head)
        return Attestation(vals, data)

    def mk_attestations(self):
        slot = get_current_slot(self.store)
        vals = tuple((slot % 4) * 4 + i for i in range(4))
        return self.mk_attestation(vals)

    def send_attestation(self, att):
        do_spec_step(self.store, att)
        self.events.append(att)


def mk_initial_states(anchor):
    anchor_state, anchor_block = anchor
    store = get_forkchoice_store(anchor_state, anchor_block)
    fc_state = get_forkchoice_state(anchor_state.freeze(), anchor_block)
    return fc_state, store


def run_tc(tc: TC):
    fc_state, store = mk_initial_states(tc.anchor)
    compare_fc_state(fc_state, store)
    for evt in tc.events:
        do_spec_step(store, evt)
        fc_state = do_abs_spec_step(fc_state, evt)
        compare_fc_state(fc_state, store)


def enumerate_stores(tc: TC, tracer=None) -> Sequence[Store]:
    anchor_state, anchor_block = tc.anchor
    if tracer is not None:
        sys.settrace(tracer)
    try:
        store = get_forkchoice_store(anchor_state, anchor_block)
    finally:
        if tracer is not None:
            sys.settrace(None)
    yield store
    for e in tc.events:
        do_spec_step(store, e, tracer)
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

def test_anchor_state():
    return TCBuilder().get_tc()


def test_simple_tick():
    builder = TCBuilder()
    builder.new_tick()
    return builder.get_tc()


def test_on_attestation():
    builder = TCBuilder()
    atts = builder.mk_attestations()
    builder.new_tick()
    builder.send_attestation(atts)
    return builder.get_tc()


def test_on_attester_slashing():
    data_1 = AttestationData(0, Checkpoint(0, 0), Checkpoint(2, 0), 0)
    data_2 = AttestationData(0, Checkpoint(1, 0), Checkpoint(2, 0), 0)
    attester_slashing = AttesterSlashing(IndexedAttestation(data_1, (1,2)), IndexedAttestation(data_2, (2,3)))
    return TC(mk_anchor(), [attester_slashing])


def test_on_block():
    builder = TCBuilder()
    for i in range(12):
        atts = builder.mk_attestations()
        builder.new_tick()
        builder.send_attestation(atts)
        builder.new_block(atts=(atts,))
    return builder.get_tc()

import sys
import inspect

class Tracer:
    def __init__(self):
        self.stats = set()
    def traceit(self, frame, event, arg):
        function_code = frame.f_code
        function_name = function_code.co_name
        file_name = function_code.co_filename
        lineno = frame.f_lineno

        if file_name.endswith('/orig_spec.py'):
            loc = f"{function_name}:{lineno}"
            self.stats.add(f"{loc}")

        return self.traceit

    def get_stats(self):
        return frozenset(self.stats)

    def clear(self):
        self.stats.clear()





b = TCBuilder()
for i in range(5):
    b.new_tick()
    b.new_block()
    slot = get_current_slot(b.store)
    if slot >= 2:
        b.new_block(slot - 2)

tcs = []
tcs.append(b.get_tc())
tcs.append(test_anchor_state())
tcs.append(test_simple_tick())
tcs.append(test_on_attestation())
tcs.append(test_on_attester_slashing())
tcs.append(test_on_block())

block_pred_cov = set()
code_cov = set()
tracer = Tracer()
for tc in tcs:
    for store in enumerate_stores(tc, tracer=tracer.traceit):
        sys.settrace(tracer.traceit)
        try:
            get_head(store)
        finally:
            sys.settrace(None)
        for block_root in store.blocks.keys():
            predicate_coverage = calc_block_predicate_coverage(store, block_root)
            block_pred_cov.add(predicate_coverage)
code_cov.update(tracer.get_stats())
tracer.clear()

print(code_cov)
print(block_pred_cov)
covered_lines = sorted(set([int(c.split(':')[1]) for c in code_cov]))

with open('/fc_comp_test/orig_spec.py', 'r') as f:
    code_lines = f.readlines()

print("missing lines")
for lno in sorted(set(range(1, len(code_lines)+1)).difference(covered_lines)):
    line = code_lines[lno - 1][:-1]
    if line.strip() not in {'', '(', ')', '[', ']', '{', '}'}:
        print(lno, line)

for c in block_pred_cov:
    print(c)