# %%

all_shapes = ['circle', 'square', 'triangle', 'diamond']
all_textures = ['plain', 'dots', 'stripes', 'checkered']

all_objs = []
for s in all_shapes:
  for t in all_textures:
    all_objs.append(f'{s}_{t}')

all_pairs = []
for i in range(len(all_objs)):
  for j in range(len(all_objs)):
    if i != j:
      all_pairs.append((all_objs[i], all_objs[j]))
# len(all_pairs) # 240

# %% Rules to functions
def get_shape(obj): return obj.split('_')[0]

def get_texture(obj): return obj.split('_')[1]

def sameAndMore (obj_arr):
  shapeMatch = all_shapes.index(get_shape(obj_arr[0])) + all_shapes.index(get_shape(obj_arr[1])) == 3
  textureMatch = True
  return ( shapeMatch and textureMatch )

sum([sameAndMore([el[0], el[1]]) for el in all_pairs])

# %%
def is_same_shape (obj_arr):
  return (
    get_shape(obj_arr[0]) == get_shape(obj_arr[1])
  )

sum([is_same_shape([el[0], el[1]]) for el in all_pairs])

def plain_diff_shapes (obj_arr):
  return (
    get_shape(obj_arr[0]) != get_shape(obj_arr[1]) and
    get_texture(obj_arr[0]) == 'plain'
  )

def complex_rule (obj_arr):
  return (
    get_shape(obj_arr[0]) in ('circle', 'square') and
    get_texture(obj_arr[1]) in ('plain', 'dots') and
    get_shape(obj_arr[1]) != 'circle'
  )

def capSet(obj_arr):
  shapeMatch = all_shapes.index(get_shape(obj_arr[0])) + all_shapes.index(get_shape(obj_arr[1])) == 3
  textureMatch = all_textures.index(get_texture(obj_arr[0])) >= all_textures.index(get_texture(obj_arr[1]))
  # textureMatch = all_textures.index(get_texture(obj_arr[0])) + all_textures.index(get_texture(obj_arr[1])) == 3
  return (
    # (shapeMatch and not textureMatch) or (not shapeMatch and textureMatch)
    shapeMatch and textureMatch
    # all_shapes.index(get_shape(obj_arr[0])) + all_shapes.index(get_shape(obj_arr[1])) == 3 and
    # all_textures.index(get_texture(obj_arr[0])) + all_textures.index(get_texture(obj_arr[1])) == 3
  )
