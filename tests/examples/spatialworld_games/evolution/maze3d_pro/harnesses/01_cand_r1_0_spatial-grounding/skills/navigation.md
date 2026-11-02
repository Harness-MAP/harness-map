### Navigation Strategy (follow this priority order every step):
1. **Gold exit dead ahead?** Walk into it. The GROUNDED STATE block gives the exact cell distance; `move_forward` until you are on it. Nothing else matters while the exit is in sight.
2. **Cyan stair dead ahead?** Walk onto it, then try `climb_up`. If the frame does not change, try `climb_down`. Floors are the usual reason an exit cannot be found on foot.
3. **Never pick an action the block lists as DEAD.** Those were proven by trying them: the frame came back byte-identical, so the world did not move. Repeating one is a guaranteed wasted step, and you only have 100.
4. **Prefer an UNEXPLORED bearing over an OPEN one.** OPEN means you have already walked that edge; the exit is not behind you. The block names the exact turn that aims you at each bearing.
5. **If the way ahead is proven to be a wall, turn — do not keep pressing forward.** One turn costs one step and always changes the view; a hundred blocked forwards cost a hundred steps and change nothing.
6. **Walk corridors with `small` steps.** It keeps the tracked position exact, which is what makes the wall map and the UNEXPLORED labels trustworthy. Use `medium`/`large` only down a corridor you can already see is long and empty.
7. **If the block says this exact view has been rendered many times, you are in a loop.** Break it by turning toward the bearing with the fewest visits, then committing to several moves in that direction instead of re-deciding every step.
8. **The exit can be on any floor.** If you have swept a floor and the block shows most bearings OPEN and already-visited, the exit is elsewhere: go back to a cyan stair and change floor.
