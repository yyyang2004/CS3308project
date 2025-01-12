import csv
import numpy as np
import pandas as pd
from collections import defaultdict

class Item:
    def __init__(self, sku_code, dimensions):
        self.sku_code = sku_code
        self.dimensions = tuple(int(dim * 10) for dim in dimensions)  # 放大10倍并转换为整数
        self.position = (-1,-1,-1)
        self.is_placed = False


class container_class:
    def __init__(self,x,y,z):
        self.H=z
        self.L=x
        self.W=y
        self.volume=x*y*z
        self.height_map=np.zeros((x,y),dtype=np.int16)
        self.utilization=0
        self.placed_items=[]

    def find_fit_location(self, item):
        """允许底面有高度差异的放置位置查找"""
        width, depth, height = item.dimensions[0],item.dimensions[1],item.dimensions[2]
        rows, cols = self.height_map.shape
        
        if width > rows or depth > cols:
            return None
        
        best_location = None
        min_height_diff = float('inf')
        min_base_height = float('inf')
        
        # 遍历所有可能的放置位置
        for x in range(rows - width + 1):
            for y in range(cols - depth + 1):
                # 获取当前区域
                region = self.height_map[x:x+width, y:y+depth]
                base_height = np.max(region)  # 使用区域最大高度作为基准
                
                if base_height + height > self.H:  # 检查是否超出容器高度
                    continue
                    
                # 计算区域内的高度差异
                height_diff = np.max(region) - np.min(region)
                
                # 选择高度差异最小且基准高度最低的位置
                if height_diff < min_height_diff or (height_diff == min_height_diff and base_height < min_base_height):
                    min_height_diff = height_diff
                    min_base_height = base_height
                    best_location = [x, y, base_height]
        
        return best_location
    
    def update_height_map(self, item):
        width, depth, height = item.dimensions
        x, y, z = item.position
        self.placed_items.append(item)
        self.height_map[x:x+width, y:y+depth] = np.maximum(
            self.height_map[x:x+width, y:y+depth],
            z + height
        )
        lx,ly,lz=item.dimensions
        self.utilization+=(lx*ly*lz)/self.volume

    def check_collision(self, x, y, z, width, depth, height):
        for placed_item in self.placed_items:
            px, py, pz = placed_item.position
            pw, pd, ph = placed_item.dimensions
            if (x < px + pw and x + width > px and  # 检查 x 轴重叠
                y < py + pd and y + depth > py and  # 检查 y 轴重叠
                z < pz + ph and z + height > pz):   # 检查 z 轴重叠
                return True
        return False

    def check_suspension(self, x, y, width, depth, z):
        if z == 0:
            return False
        
        bottom_region = self.height_map[x:x+width, y:y+depth]
        support_area = np.sum(bottom_region >= z)
        total_area = width * depth
        support_ratio = support_area / total_area
        
        return support_ratio < 0.5


def read_orders_from_csv(file_path):
    df = pd.read_csv(file_path)
    orders = defaultdict(list)
    
    for _, row in df.iterrows():
        order_id = row['sta_code']
        sku_code = row['sku_code']
        length, width, height = row['长(CM)'], row['宽(CM)'], row['高(CM)']
        quantity = row['qty']
        
        for _ in range(quantity):
            item = Item(sku_code, (length, width, height))
            orders[order_id].append(item)
            
    return orders



def greedy_packing(container_sizes, items):
    containers = []
    
    for item in sorted(items, key=lambda x: x.dimensions[0] * x.dimensions[1] * x.dimensions[2], reverse=True):
        placed = False
        
        if len(containers)!=0:
            for container in containers:
                location=container.find_fit_location(item)
                if location:
                    x, y, z = location
                    width, depth, height = item.dimensions
                    if x + width <= container.L and y + depth <= container.W and z + height <= container.H:
                        if container.check_collision(x, y, z, width, depth, height):
                            continue

                        if container.check_suspension(x, y, width, depth, z):
                            continue
                    item.position=location
                    container.update_height_map(item)
                    item.is_placed=True
                    placed=True
                    break
                        
        
        if not placed:
            # If not placed, create a new container of the smallest size that can fit the item
            for container_size in container_sizes:
                if (item.dimensions[0] <= container_size[0] and
                    item.dimensions[1] <= container_size[1] and
                    item.dimensions[2] <= container_size[2]):
                    new_container=container_class(container_size[0],container_size[1],container_size[2])
                    containers.append(new_container)
                    location=new_container.find_fit_location(item)
                    item.position=location
                    item.is_placed = True
                    new_container.update_height_map(item)
                    print(f"Placed item {item.sku_code} in new container of size {container_size}")
                    break
            if not item.is_placed:
                print(f"Item {item.sku_code} with dimensions {item.dimensions} could not be placed in any container.")

    return containers

def calculate_utilization(containers):
    sum=0
    i=0
    for container in containers:
        sum+=container.utilization
        i+=1
    return sum/i
    

def main():
    orders = read_orders_from_csv('task3.csv')
    container_sizes = [
        (35, 23, 13), (37, 26, 13), (38, 26, 13),
        (40, 28, 16), (42, 30, 18), (42, 30, 40),
        (52, 40, 17), (54, 45, 36)
    ]
    
    # 放大容器尺寸
    container_sizes = [(int(l * 10), int(w * 10), int(h * 10)) for l, w, h in container_sizes]
    
    sum_utilization=0
    num_order=0

    for order_id, items in orders.items():
        print(f"Processing order {order_id}")
        containers = greedy_packing(container_sizes, items)
        
        utilization = calculate_utilization(containers)
        print(f"Utilization for order {order_id}: {utilization:.2f}")
        sum_utilization+=utilization
        num_order+=1

        if (num_order%100==0):
            print(f"Average utilization of {num_order} orders: {sum_utilization/num_order}.")
            csv_file_name = "output.csv"
            with open(csv_file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(["num_order", "sum_utilization"])
                # 对sum_utilization进行格式化，保留2位小数
                formatted_utilization = "{:.2f}".format(sum_utilization/num_order)
                writer.writerow([num_order, formatted_utilization])
                
        
        # Check remaining unplaced items
        unplaced_items = [item for item in items if not item.is_placed]
        if unplaced_items:
            print(f"Order {order_id} has unplaced items: {[(item.sku_code, item.dimensions) for item in unplaced_items]}")
        else:
            print(f"All items placed for order {order_id}")

if __name__ == "__main__":
    main()
