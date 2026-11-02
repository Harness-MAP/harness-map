# The search contract

The cube in front of you is a FEW QUARTER TURNS from solved, and the episode ends the instant it
IS solved — you do not have to recognise the finish, only to walk over it. So the task is not to
run a solving method from a position you cannot see. It is to stand on every configuration close
to the one this episode started from, the closest ones first, and never to wander away from them.

Only one face of six is in the picture, so a tidy-looking front proves nothing about the other
five and no case can be read reliably. What CAN be read is the harness's own bookkeeping, which is
appended to these instructions every turn after the first as a HARNESS STATE block.

## Three rules that never change

1. **Never name a view rotation.** `rotate_cube_x_*`, `rotate_cube_y_*` and `rotate_cube_z_*`
   move only the camera. The cube is untouched, the step is gone, and nothing in this harness
   reads the picture, so a view turn can never pay for itself. Name a face turn, every turn.

2. **Turn the SAME face four times in a row, then move to a face you have not used yet.** Four
   quarter turns of one face pass through three configurations and come back to the start, which
   is the cheapest way to stand on new ground: six faces, twenty-four turns, and every
   configuration one quarter turn from the start has been visited. While the HARNESS STATE line
   `configurations ONE turn from the start already stood on` is below `12 of 12`, this rule alone
   is the whole plan.

3. **When the line `quarter-turns from where this episode started` reads 2, play the exact
   inverse of your last turn.** From two turns out, every move except that one leaves the
   neighbourhood where the solution has to be. The inverse of `rotate_left_cw` is
   `rotate_left_acw`, and the inverse of `rotate_left_acw` is `rotate_left_cw`. Your last turns
   are printed on the `turns executed most recently` line, so this needs no memory.

## Reading the menu

The MENU lists at most four quarter turns. Each is a move that shortens the distance from where
the cube is now to a configuration near the start that has not been stood on, and each is
annotated with what it lands on. Name one of them and it is played exactly as named — the harness
never substitutes a move you named from the menu.

Name something else and the step is still spent: the harness plays the first menu line instead,
and the `turns executed most recently` line will show you that it did. A refused name is not
thrown away — if it is a face turn, the harness tries it as soon as that turn becomes the best
move on the board — but a turn you chose from the menu happens now, and a turn you chose against
the menu happens late or never.

## What not to spend steps on

Do not start a long algorithm. A commutator such as `R U R' U'` is four turns that end three or
four quarter turns from where they began, which is outside the neighbourhood this search covers,
and from one visible face you cannot tell whether its precondition holds. Do not repeat a face
once its four turns are done — the fifth turn of a face is the first turn again and lands
somewhere already stood on. Do not undo a face block to "check" it: the harness already knows
which configurations have been visited and prints the count.
