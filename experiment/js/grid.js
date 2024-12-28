
const gridWorld = document.getElementById("task-grid");
let items_list = Object.keys(items);

let currentlyCarrying = null;

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

  // Check if player is moving into an item position
  if (playerCell.querySelector(".item-image")) {
    let itemKey = playerCell.querySelector(".item-image").id;
    getEl("task-info-location").innerHTML = `<img src="${items[itemKey].item_icon}" style="width: 24px; height: 24px;">`;
  } else {
    getEl("task-info-location").innerHTML = "";
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

  updatePlayerPosition();
}

// Handle item pick-up
function handleSpacePress() {
  if (currentlyCarrying) {
    return;
  }

  const playerCell = document.querySelector(
    `.grid-cell[data-x="${playerPosition.x}"][data-y="${playerPosition.y}"]`
  );

  const itemElement = playerCell.querySelector(".item-image");
  if (!itemElement) {
    return;
  }

  currentlyCarrying = itemElement.id;
  getEl(currentlyCarrying).remove();

  updatePlayerPosition();
  getEl("task-info-carrying").innerHTML = `<img src="${items[currentlyCarrying].item_icon}" style="width: 24px; height: 24px;">`;


}

function handleDropPress() {

  if (currentlyCarrying) {
    const dropX = playerPosition.x;
    const dropY = playerPosition.y;

    const currentCell = document.querySelector(`.grid-cell[data-x="${dropX}"][data-y="${dropY}"]`);

    // Check if there is already an item in the cell
    if (currentCell.querySelector(".item-image")) {
      return;
    }

    const itemElement = document.createElement("img");
    itemElement.src = items[currentlyCarrying].item_icon;
    itemElement.id = currentlyCarrying;
    itemElement.style.width = "30px";
    itemElement.style.height = "30px";
    itemElement.classList.add("item-image");
    currentCell.appendChild(itemElement);

    currentlyCarrying = null;
  }

  updatePlayerPosition();
  getEl("task-info-carrying").innerHTML = "";
}

// Event listener for keyboard input
document.addEventListener("keydown", (event) => {
  if (event.key === " ") {
    handleSpacePress();
  } else if (event.key === "d" || event.key === "D") {
    handleDropPress();
  } else {
    handleKeyPress(event);
  }
});

// Initialize the grid
createGrid();
