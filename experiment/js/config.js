let ACTIONS = 10;
getEl('task-info-actions').innerHTML = ACTIONS;

const gridSize = 15;
const COND = 'easy';
const MAXLEVEL = 6;

// Global variables
let items = [];
let transitions = [];


// Player state
let playerPosition = { x: 7, y: 7 };

// Item positions
let baseItems = [];
let shapes = ["triangle", "circle", "square", "diamond"];
let textures = ["plain", "checkered", "stripes", "dots"];
shapes.forEach(s => {
  textures.forEach(t => {
    baseItems.push(`${s}_${t}_0`);
  });
});
let usedPositions = new Set();
for (let i = 0; i < baseItems.length; i++) {
  let x, y, position;
  do {
    x = Math.floor(Math.random() * gridSize);
    y = Math.floor(Math.random() * gridSize);
    position = `${x},${y}`;
  } while (usedPositions.has(position));

  usedPositions.add(position);
  item_name = baseItems[i];
  items.push({
    'item_name': item_name,
    'x': x,
    'y': y,
    'item_level': 0,
    'item_icon': `../img/${item_name}.svg`
  });
}
// console.log(items);

// Transitions helper functions
function getShape(item) { return item.split("_")[0] }

function getTexture(item) { return item.split("_")[1] }

function getLevel(item) { return item.split("_")[2] }

function newObj(item1, item2) {
  const newLevel = Math.max(getLevel(item1), getLevel(item2)) + 1;
  if(newLevel > MAXLEVEL) {
     return ''

  } else {
    return `${getShape(item1)}_${getTexture(item2)}_${newLevel}`;
  }
}

function isSameShape(item_list) {
  return getShape(item_list[0]) === getShape(item_list[1]);
}

function isDiffShapeAndPlain(item_list) {
  return getShape(item_list[0]) !== getShape(item_list[1]) && getTexture(item_list[0]) === "plain";
}

function complexRule(item_list) {
  return (
    (getShape(item_list[0]) === 'circle' || getShape(item_list[0]) === 'square') &&
    (getTexture(item_list[1]) === 'plain' || getTexture(item_list[1]) === 'dots') &&
    getShape(item_list[1]) !== 'circle'
  );
}
