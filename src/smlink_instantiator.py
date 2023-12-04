from minzinc_gen import generate_sm_link_test_cases
from test_case_builder import TC, TCBuilder, get_current_slot, hash_tree_root, phase0, SignedBeaconBlock, Attestation

import eth2spec.utils.bls
eth2spec.utils.bls.bls_active = False


def instantiate_test_case(sources, targets):
    builder = TCBuilder()

    bounds = {}
    for j in range(len(sources)):
        src, tgt = sources[j], targets[j]
        max_bound = min([t for s, t in zip(sources, targets) if s == tgt], default=None)
        min_bound = max([t for s, t in zip(sources, targets) if s < tgt < t], default=None)
        bounds[tgt] = (min_bound, max_bound, j)
    hidden = {}
    to_hide_at_next_epoch = None

    def update_slot(target_slot):
        while get_current_slot(builder.store) < target_slot:
            builder.new_slot()
    for epoch in sorted(set(sources).union(targets)):
        match epoch:
            case 0:
                pass
            case int(n) if n in bounds:
                min_b, max_b, idx = bounds[epoch]
                src, tgt = sources[idx], epoch
                if src in hidden:
                    sb = hidden[src]
                    builder.send_block(sb)
                    del hidden[src]
                update_slot(phase0.compute_start_slot_at_epoch(n))
                boundary_block = builder.mk_block()
                builder.send_block(boundary_block)
                if min_b is not None:
                    update_slot(phase0.compute_start_slot_at_epoch(n) + 1)
                    if to_hide_at_next_epoch is not None:
                        epoch_to_hide, atts_to_hide = to_hide_at_next_epoch
                        hidden_block = builder.mk_block(atts=(atts_to_hide,))
                        hidden[epoch_to_hide] = hidden_block
                        to_hide_at_next_epoch = None
                    update_slot(phase0.compute_start_slot_at_epoch(n) + 2)
                    for i in [2,3,4,5,6]:
                        update_slot(phase0.compute_start_slot_at_epoch(n) + i)
                        atts = builder.mk_attestations()
                        builder.send_attestation(atts)
                    update_slot(phase0.compute_start_slot_at_epoch(n) + 7)
                    atts = builder.mk_attestations()
                    to_hide_at_next_epoch = (tgt, atts)
                else:
                    for i in range(8):
                        update_slot(phase0.compute_start_slot_at_epoch(n) + i)
                        if i == 1 and to_hide_at_next_epoch is not None:
                            epoch_to_hide, atts_to_hide = to_hide_at_next_epoch
                            hidden_block = builder.mk_block(parent=atts_to_hide.data.beacon_block_root, atts=(atts_to_hide,))
                            hidden[epoch_to_hide] = hidden_block
                            to_hide_at_next_epoch = None
                        if i == 7:
                            sb = builder.mk_block()
                            builder.send_block(sb)
                        atts = builder.mk_attestations()
                        builder.send_attestation(atts)
            case _:
                assert False
    builder.new_slot()
    if hidden:
        for k,sb in hidden.items():
            try:
                builder.send_block(sb)
            except Exception as e:
                pass
    return builder.get_tc()

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

cases = generate_sm_link_test_cases()


from test_case_runner import run_tc
from time import perf_counter


t0 = perf_counter()
for case in cases:
    sources = case['sources']
    targets = case['targets']
    tc = instantiate_test_case(sources, targets)
    print('tc', to_readable(tc))
    run_tc(tc)
t1 = perf_counter()
print(t1-t0)

