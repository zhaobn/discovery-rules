const gridSize = 15;
const gridWorld = document.getElementById("task-grid");

// Player state
let playerPosition = { x: 7, y: 7 };

// Item positions
const items = [
  { x: 5, y: 5, figure: "circle_plain_0" },
  { x: 10, y: 10, figure: "triangle_plain_0" },
  { x: 12, y: 3, figure: "circle_checkered_0" },
  { x: 8, y: 12, figure: "triangle_checkered_0" },
];

// Create the grid
function createGrid() {
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const cell = document.createElement("div");
      cell.classList.add("grid-cell");
      cell.dataset.x = x;
      cell.dataset.y = y;

      // Add items
      const item = items.find((item) => item.x === x && item.y === y);
      if (item) {
        const itemElement = document.createElement("div");
        const img = document.createElement("img");
        img.src = `../img/${item.figure}.svg`;
        img.style.width = "30px";
        img.style.height = "30px";
        itemElement.classList.add("item");
        itemElement.appendChild(img);
        cell.appendChild(itemElement);
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

  // Re-add the item if it's in the current position
  const item = items.find(
    (item) => item.x === playerPosition.x && item.y === playerPosition.y
  );

  if (item) {
    const itemElement = document.createElement("div");
    itemElement.classList.add("item");
    const img = document.createElement("img");
    img.src = `../img/${item.figure}.svg`; // Load the image
    img.style.width = "30px";
    img.style.height = "30px";
    img.classList.add("item-image");
    itemElement.appendChild(img);
    playerCell.appendChild(itemElement);
  }
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

  if (currentCell.querySelector(".item")) {
    currentCell.querySelector(".item").remove(); // Remove item if player steps on it
  }

  updatePlayerPosition();
}



// Event listener for keyboard input
document.addEventListener("keydown", handleKeyPress);

// Initialize the grid
createGrid();
