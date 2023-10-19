from minizinc import Instance, Model, Solver


sm_links = Model('../minizinc/SM_links.mzn')
gecode = Solver.lookup("gecode")
instance = Instance(gecode, sm_links)
instance['AE'] = 2 # anchor epoch
instance['NE'] = 5 # number of epochs, starting from AE
instance['NL'] = 3 # number of super-majority links

results = instance.solve(all_solutions=True)
for i in range(len(results)):
    print(i, "{", '"sources":', results[i, 'sources'], ",", '"targets":', results[i, 'targets'], "}")