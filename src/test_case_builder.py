from dataclasses import dataclass, replace
from typing import Any, Sequence, Set, Tuple, Dict, Optional


from eth2spec.test.helpers import genesis, block as block_helpers, attestations as attestations_helper

from eth2spec.phase0 import minimal as phase0
from eth2spec.phase0.minimal import MAX_EFFECTIVE_BALANCE, get_previous_epoch
from specs.phase0_minimal_fork_choice import *

import eth2spec.utils.bls


def do_spec_step(store, evt, ignore_exceptions=False):
    try:
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
    except Exception as e:
        if not ignore_exceptions:
            raise

def mk_anchor():
    no_validators = 16
    anchor_state = genesis.create_genesis_state(phase0, [MAX_EFFECTIVE_BALANCE] * no_validators, 0)
    anchor_block = BeaconBlock(state_root=hash_tree_root(anchor_state))
    return anchor_state, anchor_block




@dataclass(eq=True)
class TC:
    anchor: Tuple[BeaconState, BeaconBlock]
    events: Sequence[int|SignedBeaconBlock|Attestation|AttesterSlashing]

    def description(self):
        res = []
        for e in self.events:
            match e:
                case int(n):
                    res.append(('tick', n))
                case SignedBeaconBlock() as sb:
                    res.append(('block', hash_tree_root(sb.message)))
                case Attestation() as a:
                    res.append(('attestation', a.data.source.epoch, a.data.target.epoch))
        return res



class TCBuilder:
    def __init__(self):
        self.anchor = mk_anchor()
        self.events = []
        self.store = get_forkchoice_store(*self.anchor)
        self.attestations = []
        self.pending_atts = []

    def get_tc(self):
        assert len(self.pending_atts) == 0
        return TC(self.anchor, self.events)

    def new_slot(self):
        slot = get_current_slot(self.store)
        tick = self.store.genesis_time + config.SECONDS_PER_SLOT*(slot - GENESIS_SLOT + 1)
        do_spec_step(self.store, tick)
        self.events.append(tick)
        if self.pending_atts:
            atts = self.pending_atts[:]
            self.pending_atts.clear()
            for a in atts:
                self.send_attestation(a)

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
        elif isinstance(ref, bytes) and len(ref) == 32:
            return ref
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
            for a in atts:
                block.body.attestations.append(a)
        else:
            st = state.copy()
            phase0.process_slots(st, slot)
            selected_atts = []
            for a in self.attestations:
                assert isinstance(a, phase0.Attestation)
                if (a.data.source == st.current_justified_checkpoint
                            or a.data.source == st.previous_justified_checkpoint) \
                        and (a.data.slot + phase0.MIN_ATTESTATION_INCLUSION_DELAY <= slot <= a.data.slot + phase0.SLOTS_PER_EPOCH)\
                        and (a.data.target.epoch == get_current_epoch(st) or a.data.target.epoch == get_previous_epoch(st)):
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

    def get_curr_state(self, root=None, slot=None):
        if root is None:
            root = self._resolve_block_ref(None)
        if slot is None:
            slot = get_current_slot(self.store)
        state = self.store.block_states[root].copy()
        if state.slot < slot:
            phase0.process_slots(state, slot)
        return state


    def mk_single_attestation(self, vi, block_ref=None):
        block_root = self._resolve_block_ref(block_ref)
        state = self.get_curr_state(root=block_root)
        committee, index, slot = phase0.get_committee_assignment(state, get_current_epoch(state), vi)

        epoch = phase0.get_current_epoch(state)
        current_epoch_start_slot = phase0.compute_start_slot_at_epoch(epoch)
        if slot < current_epoch_start_slot:
            epoch_boundary_root = phase0.get_block_root(state, phase0.get_previous_epoch(state))
        elif slot == current_epoch_start_slot:
            epoch_boundary_root = block_root
        else:
            epoch_boundary_root = phase0.get_block_root(state, epoch)

        if slot < current_epoch_start_slot:
            source_epoch = state.previous_justified_checkpoint.epoch
            source_root = state.previous_justified_checkpoint.root
        else:
            source_epoch = state.current_justified_checkpoint.epoch
            source_root = state.current_justified_checkpoint.root

        attestation_data = phase0.AttestationData(
            slot=slot,
            index=index,
            beacon_block_root=block_root,
            source=phase0.Checkpoint(epoch=source_epoch, root=source_root),
            target=phase0.Checkpoint(epoch=phase0.compute_epoch_at_slot(slot), root=epoch_boundary_root),
        )


        committee_size = len(committee)
        aggregation_bits = phase0.Bitlist[phase0.MAX_VALIDATORS_PER_COMMITTEE](*([0] * committee_size))
        attestation = phase0.Attestation(
            aggregation_bits=aggregation_bits,
            data=attestation_data,
        )
        attestations_helper.fill_aggregate_attestation(phase0, state, attestation, signed=eth2spec.utils.bls.bls_active,
                                                       filter_participant_set=lambda participants: participants & {vi})
        return attestation

    def mk_attestations(self):
        head_root = self._resolve_block_ref(None)
        slot = get_current_slot(self.store)
        state = self.get_curr_state(slot=slot,root=head_root)

        block_root = head_root
        epoch = phase0.get_current_epoch(state)
        current_epoch_start_slot = phase0.compute_start_slot_at_epoch(epoch)
        if slot < current_epoch_start_slot:
            epoch_boundary_root = phase0.get_block_root(state, phase0.get_previous_epoch(state))
        elif slot == current_epoch_start_slot:
            epoch_boundary_root = block_root
        else:
            epoch_boundary_root = phase0.get_block_root(state, epoch)

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
        attestations_helper.fill_aggregate_attestation(phase0, state, attestation, signed=eth2spec.utils.bls.bls_active, filter_participant_set=None)

        atts = attestation
        return atts

    def send_attestation(self, att):
        if att.data.slot + 1 <= get_current_slot(self.store):
            do_spec_step(self.store, att)
            self.attestations.append(att)
            self.events.append(att)
        else:
            self.pending_atts.append(att)

    def send_slashing(self, slashing):
        do_spec_step(self.store, slashing)
        self.events.append(slashing)


