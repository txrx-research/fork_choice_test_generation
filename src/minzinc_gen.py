from minizinc import Instance, Model, Solver


def generate_sm_link_test_cases():
    sm_links = Model('../minizinc/SM_links.mzn')
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, sm_links)
    instance['AE'] = 0 # anchor epoch
    instance['NE'] = 5 # number of epochs, starting from AE
    instance['NL'] = 4 # number of super-majority links

    results = instance.solve(all_solutions=True)
    res = []
    for i in range(len(results)):
        res.append({'sources':  results[i, 'sources'], 'targets': results[i, 'targets'] })
    return res

if __name__ == "__main__":
    for r in generate_sm_link_test_cases():
        print(r)