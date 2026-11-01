Movement rules:
- You can move in 6 directions: up, down, left, right, forward, backward
- Cannot make 180-degree turns (reverse direction)
- Each move advances the snake one cell

Control mapping (as shown on the game screen):
- move_right(): Move right (+X direction) - Look for the "R" letter on screen
- move_left(): Move left (-X direction) - Look for the "L" letter on screen
- move_up(): Move up (-Y direction) - Look for the "U" letter on screen
- move_down(): Move down (+Y direction) - Look for the "D" letter on screen
- move_forward(): Move forward (-Z direction) - Look for the "W" letter on screen
- move_backward(): Move backward (+Z direction) - Look for the "S" letter on screen

Observation tips:
- The game shows direction indicators (letters) for valid moves
- Look at the on-screen letters to identify available directions:
  - "L" = left, "R" = right, "U" = up, "D" = down
  - "W" = forward, "S" = backward
- Indicators only show for moves that won't cause immediate collision
