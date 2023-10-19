from minizinc import Instance, Model, Solver


sm_links = Model('../minizinc/SM_links.mzn')
gecode = Solver.lookup("gecode")
instance = Instance(gecode, sm_links)
instance['AE'] = 2
instance['NE'] = 5
instance['NL'] = 3

results = instance.solve(all_solutions=True)
for i in range(len(results)):
    print(i, "{", '"sources":', results[i, 'sources'], ",", '"targets":', results[i, 'targets'], "}")