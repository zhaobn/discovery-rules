
const gridWorld = document.getElementById("task-grid");

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
      const item = items.find((item) => item.x === x && item.y === y);
      if (item) {
        cell.appendChild(drawItem(item.item_name, item.item_icon));
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

// Helper functions
function drawItem(itemId, itemIcon, type = 'grid') {
  const itemElement = document.createElement("img");
  itemElement.src = itemIcon;
  itemElement.id = itemId;
  itemElement.classList.add("item-image");
  itemElement.style.width = (type=='grid')? "30px" : "20px";
  itemElement.style.height = (type=='grid')? "30px" : "20px";

  return itemElement;

}
function nameToIcon (itemName, size='24px') {
  let item = items.find((item) => item.item_name === itemName);
  if (item) {
    return `<img src="${item.item_icon}" style="width:${size}; height:${size};">`;

  } else {
    return itemName
  }
}

// Check transitions
function updateTransitions(carriedItem, targetItem) {
  const hintDiv = getEl("task-info-hint");
  let heldItemIcon = nameToIcon(carriedItem);
  let targetItemIcon = nameToIcon(targetItem);
  let yieldItemIcon = '?'

  let seenTransitions = new Set(transitions.map(record => `${record.held}-${record.target}`));

  if (seenTransitions.has(`${carriedItem}-${targetItem}`)) {
    let record = transitions.find(record => record.held === carriedItem && record.target === targetItem);
    if (record.yield.length < 1) {
      yieldItemIcon = '❌';
    } else {
      yieldItemIcon = nameToIcon(record.yield);
    }
  }

  hintDiv.innerHTML =`${heldItemIcon} + ${targetItemIcon} = ${yieldItemIcon}`;
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
    let itemId = playerCell.querySelector(".item-image").id;
    getEl("task-info-location").innerHTML = nameToIcon(itemId);

    // Check if a transition if possible
    if (currentlyCarrying) {
      updateTransitions(currentlyCarrying, itemId);
    }

  } else {
    getEl("task-info-location").innerHTML = "";
    getEl('task-info-hint').innerHTML = '';
  }
}

// Handle player movement
function handleKeyPress(event) {
  const { x, y } = playerPosition;

  if (event.key === "ArrowUp" && y > 0) playerPosition.y--;
  if (event.key === "ArrowDown" && y < gridSize - 1) playerPosition.y++;
  if (event.key === "ArrowLeft" && x > 0) playerPosition.x--;
  if (event.key === "ArrowRight" && x < gridSize - 1) playerPosition.x++;

  updatePlayerPosition();
}

// Handle item pick-up or combination
function handleSpacePress() {
  const playerCell = document.querySelector(
    `.grid-cell[data-x="${playerPosition.x}"][data-y="${playerPosition.y}"]`
  );

  const itemElement = playerCell.querySelector(".item-image");
  if (!itemElement) {
    return;
  }

  if (currentlyCarrying) {
    combineItem(currentlyCarrying, itemElement.id);
  } else {
    currentlyCarrying = itemElement.id;
    getEl(currentlyCarrying).remove();
    getEl("task-info-carrying").innerHTML = nameToIcon(currentlyCarrying);

    updatePlayerPosition();
  }
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

    let item = items.find((item) => item.item_name === currentlyCarrying);
    currentCell.appendChild(drawItem(currentlyCarrying, item.item_icon));

    currentlyCarrying = null;

  }

  updatePlayerPosition();
  getEl("task-info-carrying").innerHTML = "";

}

function updateAction() {
  if (ACTIONS > 0) {
    ACTIONS = ACTIONS - 1
    getEl("task-info-actions").innerHTML = ACTIONS;

  } else {
    grid_done();
  }
}
function updateInventory (record) {
  const tableBody = document.getElementById('task-recipe-table').getElementsByTagName('tbody')[0];
  let row = tableBody.insertRow();
  let cellTransitions = row.insertCell(0);
  let yieldItemIcon = '❌';

  let yieldItem = record.yield;
  console.log(yieldItem);
  if (yieldItem.length > 1) {
    yieldItemIcon = nameToIcon(yieldItem);
  }

  cellTransitions.innerHTML = `${nameToIcon(record.held)} + ${nameToIcon(record.target)} = ${yieldItemIcon}`;
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });

}

// Combine items
function combineItem(heldItem, targetItem) {
  let transition = transitions.find(record => record.held === heldItem && record.target === targetItem);
  if (transition) {
    let yieldItem = transition.yield;

    if (yieldItem.length < 1) {
      getEl('task-info-hint').innerHTML = '';
      updateAction();

    } else {
      // Update the current cell
      const playerCell = document.querySelector(
        `.grid-cell[data-x="${playerPosition.x}"][data-y="${playerPosition.y}"]`
      );

      playerCell.querySelector(".item-image").remove();
      currentlyCarrying = yieldItem;

      getEl("task-info-carrying").innerHTML = nameToIcon(currentlyCarrying);

      updatePlayerPosition();
      updateAction();
    }

  } else {

    // check conditions
    let yieldItem = ''

    if (COND == 'easy' && isSameShape([heldItem, targetItem])) {
      yieldItem = newObj(heldItem, targetItem);
    }

    if (COND == 'medium' && isDiffShapeAndPlain([heldItem, targetItem])) {
      yieldItem = newObj(heldItem, targetItem);
    }

    if (COND == 'hard' && complexRule([heldItem, targetItem])) {
      yieldItem = newObj([heldItem, targetItem])
    }

    let record = {'held': heldItem, 'target': targetItem, 'yield': yieldItem}
    transitions.push(record);

    if (yieldItem.length > 0) {

      // create new item
      let itemList = items.map(item => item.item_name);
      if (itemList.indexOf(yieldItem) < 0) {
        items.push({
          'item_name': yieldItem,
          'x': playerPosition.x,
          'y': playerPosition.y,
          'item_level': getLevel(yieldItem),
          'item_icon': `../img/${yieldItem}.svg`
        });
      }

      // Update the current cell
      const playerCell = document.querySelector(
        `.grid-cell[data-x="${playerPosition.x}"][data-y="${playerPosition.y}"]`
      );

      playerCell.querySelector(".item-image").remove();
      currentlyCarrying = yieldItem;

      getEl("task-info-carrying").innerHTML = nameToIcon(currentlyCarrying);

      updatePlayerPosition();
      updateAction();

    } else {
      getEl('task-info-hint').innerHTML = '';
      updateAction();
    }

    updateInventory(record);

  }

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
