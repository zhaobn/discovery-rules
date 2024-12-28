
const gridWorld = document.getElementById("task-grid");
let items_list = Object.keys(items);

// Create the grid
function createGrid() {
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const cell = document.createElement("div");
      cell.classList.add("grid-cell");
      cell.dataset.x = x;
      cell.dataset.y = y;

      // Add items
      const itemKey = items_list.find((key) => items[key].x === x && items[key].y === y);
      if (itemKey) {
        const item = items[itemKey];
        const img = document.createElement("img");
        img.src = item.item_icon;
        img.id = itemKey;
        img.style.width = "30px";
        img.style.height = "30px";
        img.classList.add("item-image");
        cell.appendChild(img);
      }

      // Add player to initial position
      if (x === playerPosition.x && y === playerPosition.y) {
        const player = document.createElement("div");
        player.classList.add("player");
        cell.appendChild(player);
      }

      gridWorld.appendChild(cell);
    }
  }
}

// Update player position
function updatePlayerPosition() {
  document.querySelectorAll(".player").forEach((player) => player.remove());

  const playerCell = document.querySelector(
    `.grid-cell[data-x="${playerPosition.x}"][data-y="${playerPosition.y}"]`
  );

  const playerElement = document.createElement("div");
  playerElement.classList.add("player");
  playerCell.appendChild(playerElement);

}

// Handle player movement
function handleKeyPress(event) {
  const { x, y } = playerPosition;

  if (event.key === "ArrowUp" && y > 0) playerPosition.y--;
  if (event.key === "ArrowDown" && y < gridSize - 1) playerPosition.y++;
  if (event.key === "ArrowLeft" && x > 0) playerPosition.x--;
  if (event.key === "ArrowRight" && x < gridSize - 1) playerPosition.x++;

  // Check if player is moving into an item position
  const currentCell = document.querySelector(
    `.grid-cell[data-x="${playerPosition.x}"][data-y="${playerPosition.y}"]`
  );

  updatePlayerPosition();
}



// Event listener for keyboard input
document.addEventListener("keydown", handleKeyPress);

// Initialize the grid
createGrid();
