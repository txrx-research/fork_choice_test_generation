import minizinc
from minizinc import Instance, Model, Solver
from specs.phase0_minimal_fork_choice import *
from eth2spec.phase0 import minimal as phase0
from test_case_builder import TC, TCBuilder
from test_case_runner import run_tc

import random

import eth2spec.utils.bls
eth2spec.utils.bls.bls_active = False


block_tree = Model('../minizinc/Block_tree.mzn')
gecode = Solver.lookup("gecode")
instance = Instance(gecode, block_tree)
instance['NB'] = 5
instance['MC'] = 2
instance['MD'] = 5
instance['MW'] = 3


results = instance.solve(all_solutions=True)
cases = []
for i in range(len(results)):
    cases.append(results[i, 'parent'])

rnd = random.Random(123456798)

print("instantiating")
tcs = []
for case in cases:
    print(case)
    builder = TCBuilder()
    for i in range(8):
        builder.new_slot()
    blocks = {}
    for i, p in enumerate(case):
        slot = get_current_slot(builder.store)
        if i == 0:
            sb = builder.mk_block()
        else:
            parent = blocks[p]
            sb = builder.mk_block(parent=parent)
        blocks[i] = hash_tree_root(sb.message)
        builder.send_block(sb)

        curr_state = builder.get_curr_state()
        block_roots = list(builder.store.blocks.keys())
        for idx in range(phase0.get_committee_count_per_slot(curr_state, get_current_epoch(curr_state))):
            committee = phase0.get_beacon_committee(curr_state, curr_state.slot, idx)
            for vi in committee:
                block_root = rnd.choice(block_roots)
                att = builder.mk_single_attestation(vi, block_ref=block_root)
                builder.send_attestation(att)
                if rnd.randrange(0, 5) == 0:
                    bad_block_root = rnd.choice(list(set(block_roots) - {block_root}))
                    data_1 = att.data
                    data_2 = att.data.copy()
                    data_2.beacon_block_root = bad_block_root
                    attester_slashing = AttesterSlashing(
                        attestation_1=phase0.IndexedAttestation(attesting_indices=[vi], data=data_1),
                        attestation_2=phase0.IndexedAttestation(attesting_indices=[vi], data=data_2))
                    assert phase0.is_slashable_attestation_data(
                        attester_slashing.attestation_1.data, attester_slashing.attestation_2.data)
                    builder.send_slashing(attester_slashing)

        builder.new_slot()
    tc = builder.get_tc()
    print(tc.description())
    tcs.append(tc)



from mutation_operators import mutate_tc


print("running")
for tc in tcs:
    print(tc.description())
    run_tc(tc)
    print("  mutating")
    for tc_ in mutate_tc(rnd, tc, 10):
        print("  ", tc_.description())
        run_tc(tc_, ignore_exceptions=True)


