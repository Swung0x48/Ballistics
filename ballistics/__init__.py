"""Ballistics: a static analyzer for Ballance maps.

Input: a map's static collision geometry. Output: ball trajectories worth trying, each with a proof that it is
feasible under known physics rules. The game engine is not part of the search; it is the source the physics rules
are calibrated against and, afterwards, the judge of a proposed trajectory."""
