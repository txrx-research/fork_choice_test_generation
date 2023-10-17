from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Sequence, Set, Dict, Mapping, Tuple, Optional, Callable, List as PyList

from utils import hash_tree_root

class uint64(int):
    pass


class Epoch(uint64):
    pass


class Slot(uint64):
    pass


class Gwei(uint64):
    pass


class Root(int):
    pass


class ValidatorIndex(uint64):
    pass


@dataclass(eq=True, frozen=True)
class Checkpoint:
    epoch: Epoch
    root: Root


@dataclass(eq=True,frozen=True)
class BeaconBlockBody:
    # randao_reveal: BLSSignature
    # eth1_data: Eth1Data  # Eth1 data vote
    # graffiti: Bytes32  # Arbitrary data
    # Operations
    # proposer_slashings: List[ProposerSlashing, MAX_PROPOSER_SLASHINGS]
    # attester_slashings: List[AttesterSlashing, MAX_ATTESTER_SLASHINGS]
    attestations: Sequence[Attestation]
    # deposits: List[Deposit, MAX_DEPOSITS]
    # voluntary_exits: (List[SignedVoluntaryExit, MAX_VOLUNTARY_EXITS]


@dataclass(eq=True,frozen=True)
class BeaconBlock:
    parent_root: Root
    state_root: Root
    slot: Slot
    body: BeaconBlockBody


@dataclass(eq=True,frozen=True)
class SignedBeaconBlock:
    message: BeaconBlock


@dataclass(eq=True,frozen=True)
class AttestationData:
    slot: Slot
    source: Checkpoint
    target: Checkpoint
    beacon_block_root: Root


@dataclass(eq=True,frozen=True)
class Attestation:
    attesting_indices: Sequence[ValidatorIndex]
    data: AttestationData


@dataclass(eq=True,frozen=True)
class IndexedAttestation:
    data: AttestationData
    attesting_indices: Sequence[ValidatorIndex]


@dataclass(eq=True,frozen=True)
class PendingAttestation:
    attesting_indices: Sequence[ValidatorIndex]
    data: AttestationData
    inclusion_delay: Slot
    #proposer_index: ValidatorIndex


@dataclass(eq=True)
class Validator:
    slashed: bool
    effective_balance: Gwei

    def freeze(self) -> IValidator:
        return IValidator(self.slashed, self.effective_balance)


@dataclass(eq=True,frozen=True)
class IValidator:
    slashed: bool
    effective_balance: Gwei

    def unfreeze(self) -> Validator:
        return Validator(self.slashed, self.effective_balance)

@dataclass(eq=True)
class BeaconState:
    genesis_time: uint64
    slot: Slot
    validators: Sequence[Validator]
    block_roots: PyList[SLOTS_PER_HISTORICAL_ROOT]
    current_epoch_attestations: Sequence[PendingAttestation]
    previous_epoch_attestations: Sequence[PendingAttestation]
    justification_bits: Sequence[bool]
    previous_justified_checkpoint: Checkpoint
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint

    def copy(self) -> BeaconState:
        return replace(self)

    def freeze(self) -> IBeaconState:
        return IBeaconState(
            self.genesis_time, self.slot,
            tuple(v.freeze() for v in self.validators),
            self.current_justified_checkpoint, self.finalized_checkpoint)


@dataclass(eq=True,frozen=True)
class IBeaconState:
    genesis_time: uint64
    slot: Slot
    validators: Sequence[IValidator]
    current_justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint

    def copy(self) -> IBeaconState:
        return replace(self)

    def unfreeze(self) -> BeaconState:
        return BeaconState(
            self.genesis_time, self.slot,
            [v.unfreeze() for v in self.validators],
            self.current_justified_checkpoint, self.finalized_checkpoint)


@dataclass(eq=True,frozen=True)
class AttesterSlashing:
    attestation_1: IndexedAttestation
    attestation_2: IndexedAttestation


SECONDS_PER_SLOT = 12
SLOTS_PER_EPOCH = 4
GENESIS_SLOT = Slot(0)
GENESIS_EPOCH = Epoch(0)
MAX_EFFECTIVE_BALANCE = Gwei(32) # Gwei(2**5 * 10**9)
EFFECTIVE_BALANCE_INCREMENT = Gwei(1) # Gwei(2**0 * 10**9)
JUSTIFICATION_BITS_LENGTH = 4
SLOTS_PER_HISTORICAL_ROOT = 16


def copy(o: Any) -> Any:
    return replace(o)


def compute_epoch_at_slot(slot: Slot) -> Epoch:
    return Epoch(slot // SLOTS_PER_EPOCH)


def compute_start_slot_at_epoch(epoch: Epoch) -> Slot:
    return Slot(epoch * SLOTS_PER_EPOCH)


def get_current_epoch(state: BeaconState) -> Epoch:
    """
    Return the current epoch.
    """
    return compute_epoch_at_slot(state.slot)


def get_previous_epoch(state: BeaconState) -> Epoch:
    """`
    Return the previous epoch (unless the current epoch is ``GENESIS_EPOCH``).
    """
    current_epoch = get_current_epoch(state)
    return GENESIS_EPOCH if current_epoch == GENESIS_EPOCH else Epoch(current_epoch - 1)


def is_active_validator(validator: Validator, epoch: Epoch) -> bool:
    """
    Check if ``validator`` is active.
    """
    #return validator.activation_epoch <= epoch < validator.exit_epoch
    return True

def get_active_validator_indices(state: BeaconState, epoch: Epoch) -> Sequence[ValidatorIndex]:
    """
    Return the sequence of active validator indices at ``epoch``.
    """
    return [ValidatorIndex(i) for i, v in enumerate(state.validators) if is_active_validator(v, epoch)]


def get_total_balance(state: BeaconState, indices: Set[ValidatorIndex]) -> Gwei:
    """
    Return the combined effective balance of the ``indices``.
    ``EFFECTIVE_BALANCE_INCREMENT`` Gwei minimum to avoid divisions by zero.
    Math safe up to ~10B ETH, after which this overflows uint64.
    """
    return Gwei(max(EFFECTIVE_BALANCE_INCREMENT, sum([state.validators[index].effective_balance for index in indices])))


def get_total_active_balance(state: BeaconState) -> Gwei:
    """
    Return the combined effective balance of the active validators.
    Note: ``get_total_balance`` returns ``EFFECTIVE_BALANCE_INCREMENT`` Gwei minimum to avoid divisions by zero.
    """
    return get_total_balance(state, set(get_active_validator_indices(state, get_current_epoch(state))))


def get_block_root_at_slot(state: BeaconState, slot: Slot) -> Root:
    """
    Return the block root at a recent ``slot``.
    """
    #assert slot < state.slot <= slot + SLOTS_PER_HISTORICAL_ROOT, (slot, state.slot)
    return state.block_roots[slot % SLOTS_PER_HISTORICAL_ROOT]


def get_block_root(state: BeaconState, epoch: Epoch) -> Root:
    """
    Return the block root at the start of a recent ``epoch``.
    """
    return get_block_root_at_slot(state, compute_start_slot_at_epoch(epoch))


def weigh_justification_and_finalization(state: BeaconState,
                                         total_active_balance: Gwei,
                                         previous_epoch_target_balance: Gwei,
                                         current_epoch_target_balance: Gwei) -> None:
    previous_epoch = get_previous_epoch(state)
    current_epoch = get_current_epoch(state)
    old_previous_justified_checkpoint = state.previous_justified_checkpoint
    old_current_justified_checkpoint = state.current_justified_checkpoint

    # Process justifications
    state.previous_justified_checkpoint = state.current_justified_checkpoint
    state.justification_bits[1:] = state.justification_bits[:JUSTIFICATION_BITS_LENGTH - 1]
    state.justification_bits[0] = 0b0
    if previous_epoch_target_balance * 3 >= total_active_balance * 2:
        state.current_justified_checkpoint = Checkpoint(epoch=previous_epoch,
                                                        root=get_block_root(state, previous_epoch))
        state.justification_bits[1] = 0b1
    if current_epoch_target_balance * 3 >= total_active_balance * 2:
        state.current_justified_checkpoint = Checkpoint(epoch=current_epoch,
                                                        root=get_block_root(state, current_epoch))
        state.justification_bits[0] = 0b1

    # Process finalizations
    bits = state.justification_bits
    # The 2nd/3rd/4th most recent epochs are justified, the 2nd using the 4th as source
    if all(bits[1:4]) and old_previous_justified_checkpoint.epoch + 3 == current_epoch:
        state.finalized_checkpoint = old_previous_justified_checkpoint
    # The 2nd/3rd most recent epochs are justified, the 2nd using the 3rd as source
    if all(bits[1:3]) and old_previous_justified_checkpoint.epoch + 2 == current_epoch:
        state.finalized_checkpoint = old_previous_justified_checkpoint
    # The 1st/2nd/3rd most recent epochs are justified, the 1st using the 3rd as source
    if all(bits[0:3]) and old_current_justified_checkpoint.epoch + 2 == current_epoch:
        state.finalized_checkpoint = old_current_justified_checkpoint
    # The 1st/2nd most recent epochs are justified, the 1st using the 2nd as source
    if all(bits[0:2]) and old_current_justified_checkpoint.epoch + 1 == current_epoch:
        state.finalized_checkpoint = old_current_justified_checkpoint


def get_matching_source_attestations(state: BeaconState, epoch: Epoch) -> Sequence[PendingAttestation]:
    assert epoch in (get_previous_epoch(state), get_current_epoch(state))
    return state.current_epoch_attestations if epoch == get_current_epoch(state) else state.previous_epoch_attestations


def get_matching_target_attestations(state: BeaconState, epoch: Epoch) -> Sequence[PendingAttestation]:
    return [
        a for a in get_matching_source_attestations(state, epoch)
        #if a.data.target.root == get_block_root(state, epoch)
    ]

def get_matching_head_attestations(state: BeaconState, epoch: Epoch) -> Sequence[PendingAttestation]:
    return [
        a for a in get_matching_target_attestations(state, epoch)
        #if a.data.beacon_block_root == get_block_root_at_slot(state, a.data.slot)
    ]

def get_unslashed_attesting_indices(state: BeaconState,
                                    attestations: Sequence[PendingAttestation]) -> Set[ValidatorIndex]:
    output = set()  # type: Set[ValidatorIndex]
    for a in attestations:
        output = output.union(a.attesting_indices)
    return set(filter(lambda index: not state.validators[index].slashed, output))

def get_attesting_balance(state: BeaconState, attestations: Sequence[PendingAttestation]) -> Gwei:
    """
    Return the combined effective balance of the set of unslashed validators participating in ``attestations``.
    Note: ``get_total_balance`` returns ``EFFECTIVE_BALANCE_INCREMENT`` Gwei minimum to avoid divisions by zero.
    """
    return get_total_balance(state, get_unslashed_attesting_indices(state, attestations))


def process_justification_and_finalization(state: BeaconState) -> None:
    # Initial FFG checkpoint values have a `0x00` stub for `root`.
    # Skip FFG updates in the first two epochs to avoid corner cases that might result in modifying this stub.
    if get_current_epoch(state) <= GENESIS_EPOCH + 1:
        return
    previous_attestations = get_matching_target_attestations(state, get_previous_epoch(state))
    current_attestations = get_matching_target_attestations(state, get_current_epoch(state))
    total_active_balance = get_total_active_balance(state)
    previous_target_balance = get_attesting_balance(state, previous_attestations)
    current_target_balance = get_attesting_balance(state, current_attestations)
    weigh_justification_and_finalization(state, total_active_balance, previous_target_balance, current_target_balance)


def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    data = attestation.data
    assert data.target.epoch in (get_previous_epoch(state), get_current_epoch(state))
    assert data.target.epoch == compute_epoch_at_slot(data.slot)
    #assert data.slot + MIN_ATTESTATION_INCLUSION_DELAY <= state.slot <= data.slot + SLOTS_PER_EPOCH
    #assert data.index < get_committee_count_per_slot(state, data.target.epoch)

    #committee = get_beacon_committee(state, data.slot, data.index)
    #assert len(attestation.aggregation_bits) == len(committee)

    pending_attestation = PendingAttestation(
        data=data,
        attesting_indices=attestation.attesting_indices,
        inclusion_delay=state.slot - data.slot,
        #proposer_index=get_beacon_proposer_index(state),
    )

    if data.target.epoch == get_current_epoch(state):
        #assert data.source == state.current_justified_checkpoint
        print(state.current_epoch_attestations)
        state.current_epoch_attestations.append(pending_attestation)
    else:
        #assert data.source == state.previous_justified_checkpoint
        state.previous_epoch_attestations.append(pending_attestation)

    # Verify signature
    assert is_valid_indexed_attestation(state, get_indexed_attestation(state, attestation))

def process_operations(state: BeaconState, body: BeaconBlockBody) -> None:
    # Verify that outstanding deposits are processed up to the maximum number of deposits
    # assert len(body.deposits) == min(MAX_DEPOSITS, state.eth1_data.deposit_count - state.eth1_deposit_index)

    def for_ops(operations: Sequence[Any], fn: Callable[[BeaconState, Any], None]) -> None:
        for operation in operations:
            fn(state, operation)

    # for_ops(body.proposer_slashings, process_proposer_slashing)
    # for_ops(body.attester_slashings, process_attester_slashing)
    for_ops(body.attestations, process_attestation)
    # for_ops(body.deposits, process_deposit)
    # for_ops(body.voluntary_exits, process_voluntary_exit)


def process_block(state: BeaconState, block: BeaconBlock) -> None:
    # process_block_header(state, block)
    # process_randao(state, block.body)
    # process_eth1_data(state, block.body)
    process_operations(state, block.body)


def process_epoch(state: BeaconState) -> None:
    process_justification_and_finalization(state)
    # process_rewards_and_penalties(state)
    # process_registry_updates(state)
    # process_slashings(state)
    # process_eth1_data_reset(state)
    # process_effective_balance_updates(state)
    # process_slashings_reset(state)
    # process_randao_mixes_reset(state)
    # process_historical_roots_update(state)
    # process_participation_record_updates(state)


def process_slot(state: BeaconState) -> None:
    # Cache state root
    previous_state_root = hash_tree_root(state)
    #state.state_roots[state.slot % SLOTS_PER_HISTORICAL_ROOT] = previous_state_root
    # Cache latest block header state root
    if state.latest_block_header.state_root == 0:
        state.latest_block_header.state_root = previous_state_root
    # Cache block root
    previous_block_root = hash_tree_root(state.latest_block_header)
    state.block_roots[state.slot % SLOTS_PER_HISTORICAL_ROOT] = previous_block_root


def process_slots(state: BeaconState, slot: Slot) -> None:
    assert state.slot < slot
    while state.slot < slot:
        process_slot(state)
        # Process epoch on the start slot of the next epoch
        if (state.slot + 1) % SLOTS_PER_EPOCH == 0:
            process_epoch(state)
        state.slot = Slot(state.slot + 1)

def state_transition(state: BeaconState, signed_block: SignedBeaconBlock, validate_result: bool=True) -> None:
    block = signed_block.message
    # Process slots (including those with no blocks) since block
    process_slots(state, block.slot)
    # Verify signature
    #if validate_result:
    #    assert verify_block_signature(state, signed_block)
    # Process block
    process_block(state, block)
    # Verify state root
    #if validate_result:
    #    assert block.state_root == hash_tree_root(state)


def get_indexed_attestation(state: BeaconState, attestation: Attestation) -> IndexedAttestation:
    """
    Return the indexed attestation corresponding to ``attestation``.
    """
    #attesting_indices = get_attesting_indices(state, attestation.data, attestation.aggregation_bits)
    attesting_indices = attestation.attesting_indices

    return IndexedAttestation(
        attesting_indices=sorted(attesting_indices),
        data=attestation.data,
        #signature=attestation.signature,
    )

def is_valid_indexed_attestation(state: BeaconState, indexed_attestation: IndexedAttestation) -> bool:
    """
    Check if ``indexed_attestation`` is not empty, has sorted and unique indices and has a valid aggregate signature.
    """
    # Verify indices are sorted and unique
    indices = indexed_attestation.attesting_indices
    if len(indices) == 0 or not list(indices) == sorted(set(indices)):
        return False
    # Verify aggregate signature
    #pubkeys = [state.validators[i].pubkey for i in indices]
    #domain = get_domain(state, DOMAIN_BEACON_ATTESTER, indexed_attestation.data.target.epoch)
    #signing_root = compute_signing_root(indexed_attestation.data, domain)
    #return bls.FastAggregateVerify(pubkeys, signing_root, indexed_attestation.signature)
    return True

def is_slashable_attestation_data(data_1: AttestationData, data_2: AttestationData) -> bool:
    """
    Check if ``data_1`` and ``data_2`` are slashable according to Casper FFG rules.
    """
    return (
        # Double vote
            (data_1 != data_2 and data_1.target.epoch == data_2.target.epoch) or
            # Surround vote
            (data_1.source.epoch < data_2.source.epoch and data_2.target.epoch < data_1.target.epoch)
    )


