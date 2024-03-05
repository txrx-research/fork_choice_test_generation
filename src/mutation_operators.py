import random
import unittest

from toolz import mapcat

from test_case_builder import TC
from eth2spec.phase0.minimal import (BeaconBlock, SignedBeaconBlock, BeaconState, Attestation, AttesterSlashing,
                                     IndexedAttestation)
from eth2spec.phase0.minimal import get_indexed_attestation
from eth2spec.utils.ssz.ssz_impl import hash_tree_root

def tc_to_tv(tc: TC):
    tv = []
    curr = 0
    for evt in tc.events:
        if isinstance(evt, int):
            curr = evt
        else:
            tv.append((curr, evt))
    return tv


def tv_to_tc(anchor, tv):
    events = []
    for t,e in tv:
        events.extend([t, e])
    return TC(anchor, events)


def sort_events(tv):
    return sorted(tv, key=lambda p: p[0])


def mut_shift_(tv, idx, delta):
    time, event = tv[idx]
    new_time = int(time) + delta
    if new_time >= 0:
        return sorted(tv[:idx] + [(new_time, event)] + tv[idx+1:], key=lambda x: x[0])


def mut_shift(tv, rnd: random.Random):
    idx = rnd.choice(range(len(tv)))
    idx_time = tv[idx][0]
    dir = rnd.randint(0, 1)
    if idx_time == 0 or dir:
        time_shift = rnd.randint(0, 6) * 3
    else:
        time_shift = -rnd.randint(0, idx_time // 3)
    return mut_shift_(tv, idx, time_shift)


def mut_drop_(tv, idx):
    return tv[:idx] + tv[idx+1:]


def mut_drop(tv, rnd: random.Random):
    idx = rnd.choice(range(len(tv)))
    return mut_drop_(tv, idx)


def mut_dup_(tv, idx, shift):
    return mut_shift_(tv + [tv[idx]], len(tv), shift)


def mutate_tc(rnd, tc, cnt):
    tc_ = tc
    for i in range(cnt):
        coin = rnd.randint(0, 1)
        if coin:
            print("  mutating tc")
            tc__ = tc
        else:
            print("  mutating tc_")
            tc__ = tc_
        tv = tc_to_tv(tc__)
        op_kind = rnd.randint(0, 2)
        if op_kind == 0:
            idx = rnd.choice(range(len(tv)))
            print(f"  dropping {idx}")
            tv_ = mut_drop_(tv, idx)
        elif op_kind == 1:
            idx = rnd.choice(range(len(tv)))
            idx_time = tv[idx][0]
            dir = rnd.randint(0, 1)
            if idx_time == 0 or dir:
                time_shift = rnd.randint(0, 6) * 3
            else:
                time_shift = -rnd.randint(0, idx_time // 3) * 3
            print(f"  shifting {idx} by {time_shift}")
            tv_ = mut_shift_(tv, idx, time_shift)
        elif op_kind == 2:
            idx = rnd.choice(range(len(tv)))
            shift = rnd.randint(0, 5) * 3
            print(f"  dupping {idx} and shifting by {shift}")
            tv_ = mut_dup_(tv, idx, shift)
        elif op_kind == 3:
            blocks = [evt.message for _, evt in tv if isinstance(evt, SignedBeaconBlock)]
            atts = [evt for _, evt in tv if isinstance(evt, Attestation)]
            block_atts = list(mapcat(lambda b: b.body.attestations, blocks))
            attestations = atts + block_atts
            if len(attestations) > 0:
                att = rnd.choice(attestations)
                data = att.data
                data2 = data.copy()
                other_block = list(set(map(lambda b: hash_tree_root(b), blocks)) - {data.beacon_block_root})[0]
                data2.beacon_block_root = other_block
                IndexedAttestation()
                AttesterSlashing()
                pass
        else:
            assert False
        tc_ = tv_to_tc(tc.anchor, tv_)
        yield tc_


class MutatorTest(unittest.TestCase):
    def test_shift(self):
        block = SignedBeaconBlock()
        tv = [(0, block)]
        self.assertEqual(mut_shift_(tv, 0, 5), [(5, block)])
        self.assertEqual(mut_shift_(tv, 0, -5), None)

    def test_drop(self):
        b0 = SignedBeaconBlock()
        b1 = SignedBeaconBlock()
        b2 = SignedBeaconBlock()
        tv = [(0, b0), (6, b1), (12, b2)]
        self.assertEqual(mut_drop_(tv, 1), [(0, b0), (12, b2)])

    def test_conversion(self):
        anchor = (BeaconState(), BeaconBlock())
        b0 = BeaconBlock()
        self.assertEqual(tc_to_tv(TC(anchor, [0, b0])), [(0, b0)])
        self.assertEqual(tv_to_tc(anchor, [(0, b0)]), TC(anchor, [0, b0]))