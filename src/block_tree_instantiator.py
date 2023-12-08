import minizinc
from minizinc import Instance, Model, Solver
from test_case_builder import TC, TCBuilder, get_current_slot, hash_tree_root, phase0, SignedBeaconBlock, Attestation
from test_case_runner import run_tc

import eth2spec.utils.bls
eth2spec.utils.bls.bls_active = False

def to_readable(tc: TC):
    res = []
    for e in tc.events:
        match e:
            case int(n):
                res.append(('tick', n))
            case SignedBeaconBlock() as sb:
                res.append(('block', hash_tree_root(sb.message)))
            case Attestation() as a:
                res.append(('attestation', a.data.source.epoch, a.data.target.epoch))
    return res


block_tree = Model('../minizinc/Block_tree.mzn')
gecode = Solver.lookup("gecode")
instance = Instance(gecode, block_tree)


results = instance.solve(all_solutions=True)
cases = []
for i in range(len(results)):
    cases.append(results[i, 'parent'])

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
        builder.new_slot()
    tc = builder.get_tc()
    print(to_readable(tc))
    run_tc(tc)
