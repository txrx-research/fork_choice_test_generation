include "globals.mzn";

int: NE = 5;
int: ML = NE-2;
int: MB = NE-2;

set of int: EPOCH = 0..NE-1;
set of int: LINK = 0..ML;
set of int: BLOCK = 0..MB;

type BS = record(EPOCH: e, EPOCH: je, EPOCH: uje);

array[LINK] of var EPOCH: sources;
array[LINK] of var EPOCH: targets;

predicate surround_vote(var LINK: a, var LINK: b) =
  sources[a] < sources[b] /\ targets[b] < targets[a];


constraint forall(l in LINK)(sources[l] < targets[l]);
constraint forall(i in LINK)(sources[i] == 0 \/ member(targets, sources[i]));
constraint strictly_increasing(targets);
constraint forall(i,j in LINK where i != j)(not surround_vote(i,j));


array[BLOCK] of var EPOCH: es;
array[BLOCK] of var EPOCH: pjes;
array[BLOCK] of var EPOCH: cjes;
array[BLOCK] of var bool: prevs;
array[BLOCK] of var bool: currs;

predicate prev_link(var BLOCK: b) = es[b] > 0 /\ exists(l in LINK)(sources[l] == pjes[b] /\ targets[l] = es[b]-1);
predicate curr_link(var BLOCK: b) = exists(l in LINK)(sources[l] = cjes[b] /\ targets[l] == es[b]);

function var EPOCH: ne(var BLOCK: b) = es[b] + 1;
function var EPOCH: npje(var BLOCK: b) = cjes[b];
function var EPOCH: ncje(var BLOCK: b) =
  if currs[b] /\ curr_link(b) then
    es[b]
  elseif prevs[b] /\ prev_link(b) then
    es[b] - 1
  else
    cjes[b]
  endif;

var BLOCK: max_block;
var BLOCK: curr_block;

constraint es[0] == 0;
constraint forall(b in BLOCK where b <= max_block)((es[b] == 0) -> (cjes[b] == 0 /\ pjes[b] == 0));
constraint forall(b in BLOCK where b < max_block /\ (b+1) in BLOCK)(ne(b) == es[b+1] /\ npje(b) == pjes[b+1] /\ ncje(b) == cjes[b+1]);

function var BS: mk_block(var BLOCK: b) =
  (e: es[b], je: cjes[b], uje: ncje(b));


constraint curr_block <= max_block;
var BS: block = mk_block(curr_block);


var EPOCH: curr_e;
var EPOCH: store_je;

function var EPOCH: get_vse(var BS: block) =
  if block.e < curr_e then block.uje else block.je endif;

constraint forall(b in BLOCK where b <= max_block)(mk_block(b).e <= curr_e);
constraint forall(b in BLOCK where b <= max_block)(get_vse(mk_block(b)) <= store_je);
constraint exists(b in BLOCK where b <= max_block)(get_vse(mk_block(b)) == store_je);



bool: store_je_eq_zero = false;
bool: block_vse_eq_store_je;
bool: prev_e_justified;
bool: block_uje_ge_store_je;
bool: block_vse_plus_two_ge_curr_e;

constraint store_je_eq_zero <-> store_je == 0;
constraint block_vse_eq_store_je <-> get_vse(block) == store_je;
constraint prev_e_justified <-> store_je == curr_e - 1;
constraint block_uje_ge_store_je <-> block.uje >= store_je;
constraint block_vse_plus_two_ge_curr_e <-> get_vse(block) + 2 >= curr_e;
