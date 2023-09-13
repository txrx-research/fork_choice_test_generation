include "globals.mzn";

int: NE = 8;

set of int: EPOCH = 0..(NE-1);
set of int: LINKS = 0..(NE-2);

array[LINKS] of var EPOCH: sources;
array[LINKS] of var EPOCH: targets;

predicate surround_vote(var LINKS: a, var LINKS: b) =
  sources[a] < sources[b] /\ targets[b] < targets[a];

constraint forall(i in LINKS)(sources[i] < targets[i]);
constraint forall(i in LINKS)(sources[i] == 0 \/ member(targets, sources[i]));
constraint strictly_increasing(targets);
constraint forall(i,j in LINKS where i != j)(not surround_vote(i,j));
