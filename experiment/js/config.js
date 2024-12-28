const ACTIONS = 10;
const gridSize = 15;

// Player state
let playerPosition = { x: 7, y: 7 };

// Item positions
let items = {};

let baseItems = [];
let shapes = ["triangle", "circle", "square", "diamond"];
let textures = ["plain", "checkered", "stripes", "dots"];
shapes.forEach(s => {
  textures.forEach(t => {
    baseItems.push(`${s}_${t}_0`);
  });
});
let usedPositions = new Set(`${playerPosition.x},${playerPosition.y}`); // player position
for (let i = 0; i < baseItems.length; i++) {
  let x, y, position;
  do {
    x = Math.floor(Math.random() * gridSize);
    y = Math.floor(Math.random() * gridSize);
    position = `${x},${y}`;
  } while (usedPositions.has(position));

  usedPositions.add(position);
  item_name = baseItems[i];
  items[item_name] = {
    'x': x,
    'y': y,
    'item_level': 0,
    'item_icon': `../img/${item_name}.svg`
  };
}
// console.log(items);

// generate transitions
const transitions = [];
