import random
import numpy as np

NUM_ITEM_MIN = 10
NUM_ITEM_MAX = 50
CONTAINER_SIZE = (100, 100, 100)

def rotate_item(item):
    """
    生成数据时随机旋转
    """
    axis = random.choice([0, 1, 2])
    way  = random.choice([0, 1]) 
    if axis == 0:
        if way == 0:
            return [item[0], item[1], item[2]] 
        else:
            return [item[0], item[2], item[1]] 
    elif axis == 1:
        if way == 0:
            return [item[2], item[0], item[1]]
        else:
            return [item[2], item[1], item[0]]
    else:
        if way == 0:
            return [item[1], item[0], item[2]]
        else:
            return [item[1], item[2], item[0]]

def split_item(item, axis):
    """
    对Item进行分割
    """
    if item[axis] <= 2:
        return None, None
    
    position = random.randint(1, item[axis] - 1)
    
    new_item_1 = item.copy()
    new_item_2 = item.copy()
    
    new_item_1[axis] = position
    new_item_2[axis] = item[axis] - position
    
    return new_item_1, new_item_2

def generate_bpp_data(num_items=0, num_items_min=NUM_ITEM_MIN, num_items_max=NUM_ITEM_MAX, container_size=CONTAINER_SIZE):
    """
    Algorithm1, 生成数据:num_itmes个item, 容器大小默认(100, 100, 100)
    """
    items = [[100, 100, 100]]
    
    if num_items == 0:
        num_items = random.randint(num_items_min, num_items_max)
    
    while len(items) < num_items:
        item = random.choices(items, weights=[np.prod(i) for i in items])[0]
        axis = random.choice([0, 1, 2])
        items.remove(item)
        
        new_item_1, new_item_2 = split_item(item, axis)

        if new_item_1 is None or new_item_2 is None:
            items.append(item)
            continue

        new_item_1 = rotate_item(new_item_1)
        new_item_2 = rotate_item(new_item_2)
        
        items.append(new_item_1)
        items.append(new_item_2)
    
    return len(items), items, container_size

"""
items, container_size = generate_bpp_data()
print(f"Generated items: {items}")
Sizes = 0
for item in items:
    Sizes += np.prod(item)
print(Sizes)
"""
