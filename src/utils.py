from typing import Any, Dict
#from beacon_chain import Root, BeaconState

objects: Dict[Any, int] = {}


def hash_tree_root(object: Any):
    global objects
    if 'BeaconState' in str(type(object)):
        object = object.freeze()

    if object not in objects:
        objects[object] = len(objects)
    return objects[object]



