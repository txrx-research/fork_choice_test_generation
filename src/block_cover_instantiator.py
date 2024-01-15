import minizinc
from minizinc import Instance, Model, Solver
#from test_case_builder import TC, TCBuilder, get_current_slot, hash_tree_root, phase0, SignedBeaconBlock, Attestation
from test_case_runner import run_tc

import eth2spec.utils.bls
eth2spec.utils.bls.bls_active = False

predicates = [
    'block_vse_eq_store_je', 'prev_e_justified', 'block_uje_ge_store_je',
    'block_vse_plus_two_ge_curr_e'
]

def enum_predicates(preds):
    if len(preds) == 0:
        yield {}
    else:
        for c in enum_predicates(preds[1:]):
            yield c | { preds[0]: True }
            yield c | { preds[0]: False }

for p in enum_predicates(predicates):
    print(p)
    block_predicates = Model('../minizinc/Block_cover2.mzn')
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, block_predicates)
    for k, v in p.items():
        instance[k] = v
    results = instance.solve()
    print(results)
