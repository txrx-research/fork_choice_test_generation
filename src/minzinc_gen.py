from minizinc import Instance, Model, Solver


def generate_sm_link_test_cases():
    sm_links = Model('../minizinc/SM_links.mzn')
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, sm_links)
    instance['AE'] = 0 # anchor epoch
    instance['NE'] = 5 # number of epochs, starting from AE
    instance['NL'] = 3 # number of super-majority links

    results = instance.solve(all_solutions=True)
    res = []
    for i in range(len(results)):
        res.append({'sources':  results[i, 'sources'], 'targets': results[i, 'targets'] })
    return res

if __name__ == "__main__":
    for r in generate_sm_link_test_cases():
        print(r)
        # sources = r['sources']
        # targets = r['targets']
        # print('-------')
        # print(list(zip(sources, targets)))
        # bounds = {}
        # for j in range(len(sources)):
        #     src, tgt = sources[j], targets[j]
        #     max_bound = min([t for s, t in zip(sources, targets) if s == tgt], default=None)
        #     min_bound = max([t for s, t in zip(sources, targets) if s < tgt < t], default=None)
        #     bounds[tgt] = (min_bound, max_bound, j)
        # hidden = set()
        # to_hide_at_next_epoch = None
        # for epoch in sorted(set(sources).union(targets)):
        #     match epoch:
        #         case 0:
        #             pass
        #         case int(n) if n in bounds:
        #             min_b, max_b, idx = bounds[epoch]
        #             src, tgt = sources[idx], epoch
        #             if src in hidden:
        #                 print('  reveal block justifying', src, 'before next epoch')
        #                 hidden.remove(src)
        #             print(f'epoch {n}')
        #             print(f'  make boundary block (optional)')
        #             if to_hide_at_next_epoch is not None:
        #                 print(f'  make hidden block for prev epoch with hidden atts {to_hide_at_next_epoch}')
        #                 to_hide_at_next_epoch = None
        #             if min_b is not None:
        #                 hidden.add(tgt)
        #                 print(f'  skip atts at slots {epoch*8}, {epoch*8+1}')
        #                 print(f'  attest slots {",".join(str(epoch*8+i) for i in [2,3,4,5,6])}')
        #                 print(f'  attest and hide slot {epoch*8+7}')
        #                 to_hide_at_next_epoch = epoch*8+7
        #             else:
        #                 print(f'  attest slots {",".join(str(epoch*8+i) for i in range(8))}')
        #                 print(f'  make block with atts')
        #         case _:
        #             assert False
        # if hidden:
        #     print(f'  reveal {hidden}')

