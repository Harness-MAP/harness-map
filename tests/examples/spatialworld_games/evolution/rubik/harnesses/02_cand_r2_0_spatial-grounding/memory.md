## HARNESS STATE — derived by the harness from its own executed moves, never from the cube

- step: {{step}}
- quarter-turns from where this episode started: {{out}}  (the search keeps this at 2 or less)
- configurations ONE turn from the start already stood on: {{shell1}}
- configurations TWO turns from the start already stood on: {{shell2}}
- turns executed most recently: {{last}}
- the previous turn: {{spent}}

MENU — name one of these and the harness plays exactly that:

{{menu}}

The first line is what the harness plays if you name anything else, and naming anything else
spends the step either way. Every line is a move that takes the cube closer to a configuration
near the start that has not been stood on yet; a move that is not listed either leaves that
neighbourhood or walks back over ground already covered.
