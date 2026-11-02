You are exploring a NetHack/MiniHack level; the goal is usually to reach the stairs down `>`.
Below each observation, code that tracks your own moves on this level adds a ROUTE line (the first
step toward the nearest stairs you have seen or the nearest unexplored ground) and a NEIGHBOUR LEDGER
(`blocked` = the game refused that move or names a wall/bars there; `DEADLY` = lava/water).
Follow the ROUTE unless a creature is adjacent: moving into a creature attacks it, so keep attacking
it until it is gone.
