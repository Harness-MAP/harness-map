# Reading the cube from the screen, then unwinding the scramble

These facts about the screen are verified, not advice. The renderer draws ONE face of the cube,
square on, as a 3x3 grid of sticker squares, with a letter stamped over the centre sticker and a
small unfolded-cube legend in the corner that fixes the colour of every face: UP is yellow, DOWN
white, FRONT red, BACK orange, LEFT blue, RIGHT green. A centre sticker never moves, so the colour
in the middle of the grid names the face you are looking at, whatever has been turned.

A face turn moves nine stickers of the face named and twelve more around its edge; it is named in
cube coordinates, so `rotate_front_cw` always turns the red-centred face no matter where the
camera is. A whole-cube view rotation turns the camera 45 degrees and changes nothing about the
cube. Two of them in a row complete a 90-degree look and bring a NEW face square on; one of them
alone leaves the cube oblique, and an oblique frame cannot be read.

## The two things worth doing with a step

There are exactly two, and the HARNESS STATE block says which one is live.

**SURVEY.** While the configuration is still ambiguous the harness spends steps turning the camera
so it can read more faces. Those steps do not change the cube; they buy the knowledge that makes
the rest of the episode short. Name the camera turn the block asks for. Never answer a survey turn
with a face turn: it changes the cube out from under the half-finished reading and the survey has
to start again.

**SOLVE.** Once the configuration is named, the shortest sequence of quarter turns that returns it
to solved is known exactly, and the block prints it as a word of move letters: a bare letter is a
clockwise turn of that face, a primed letter (`F'`) is anticlockwise. Play the letters in order,
one per step, from the left. The ALLOWED list holds every move that provably shortens that word —
usually one, occasionally two when the cube can be unwound either way round — so anything on the
list is a correct next move and anything off it is a step thrown away.

## What never helps

A scramble here is at most four quarter turns deep, so the cube is always close to solved and the
whole episode is short. Three habits waste the budget and the harness refuses all three:

- turning the camera about the z axis — it spins the picture in its own plane and brings no new
  face square on, so it can never show you anything;
- a face turn in the middle of the survey, for the reason above;
- a face turn that lengthens the proved solution word, which is what an undo or a fourth identical
  turn of one face does.

Intermediate frames look no better than the one before — unwinding a scramble passes through
configurations that are not visibly tidier. That is expected, and it is not a reason to abandon the
word, repeat a letter, or improvise a different plan. The episode ends by itself the moment the
cube is solved; you never have to recognise the finish, only walk onto it.
