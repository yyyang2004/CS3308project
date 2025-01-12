from generate import *
import gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

L = 100
W = 100
H = 100

#旋转表示
Rotation = {
    "xyz": 0,
    "xzy": 1,
    "yxz": 2,
    "yzx": 3,
    "zxy": 4,
    "zyx": 5
}

#num_items, items, container_size = generate_bpp_data()

class Position:  
    """
    在容器中的位置[x, y, z]
    """
    UNPLACED = [-1,-1,-1]
    
    def __init__(self, position):
        self.position = position
        
    def __is_equal__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    @property
    def x(self): return self.position[0]
    @property
    def y(self): return self.position[1]
    @property
    def z(self): return self.position[2]


class Item:  
    """
    item的属性(x, y, z, rotation, placed, position, 当前在xyz轴的长度(区别于本身属性长度))
    """
    def __init__(self, item):
        self.x = item[0]
        self.y = item[1]
        self.z = item[2]
        self.rotation = 0  #默认xyz
        self.placed = False
        self.position = Position(Position.UNPLACED)
    
    def set_position(self, position_):
        self.position = position_
    
    def set_rotation(self, rotation_):
        self.rotation = rotation_
    
    def set_placed(self, placed_):
        self.placed = placed_
    
    @property   
    def vol(self):
        return self.x * self.y * self.z
    
    @property
    def pos(self):
        return self.position.position  # 返回物品的位置
    
    @property
    def current_state(self):
        item = [self.x, self.y, self.z]
        rotation_map = {
            0: item,  # "xyz"
            1: [item[0], item[2], item[1]],  # "xzy"
            2: [item[1], item[0], item[2]],  # "yxz"
            3: [item[1], item[2], item[0]],  # "yzx"
            4: [item[2], item[0], item[1]],  # "zxy"
            5: [item[2], item[1], item[0]]  # "zyx"
        }
        return rotation_map[self.rotation]

class BoxEnv(gym.Env):
    """
    box state, placing stratagy and box env
    """
    metadata = {'render.modes': ['human', 'print']} # 渲染模式

    def __init__(self, num_items=0, container_size=CONTAINER_SIZE, num_items_min=NUM_ITEM_MIN, num_items_max=NUM_ITEM_MAX):
        super(BoxEnv, self).__init__()
        
        self.L, self.W, self.H = container_size
        self.height_map = np.zeros((self.L, self.W), dtype=np.int16) # 存储容器在xy平面上的高度, 也即目前容器的状态
        self.items = []
        self.placed_items = []
        self.num_items = 0
        
        self.num_items, items, _ = generate_bpp_data(num_items, num_items_min, num_items_max, container_size)
        for item in items:
            self.items.append(Item(item))
        
        self.action_space = spaces.Discrete(self.num_items)
        self.observation_space = spaces.Dict({
            "height_map": spaces.Box(0, self.H, shape=(self.L, self.W), dtype=np.int16),
            "mask": spaces.MultiBinary(self.num_items) # mask标记哪些item可用(1:可用, 0:不可用)
        })
        self.current_step = 0
        self.fail_time = 0
        self.available_mask = np.ones(self.num_items, dtype=bool)
    
    def reset(self, change_items=False, seed=None, num_items=0, container_size=CONTAINER_SIZE, num_items_min=NUM_ITEM_MIN, num_items_max=NUM_ITEM_MAX):
        if seed is not None:
            np.random.seed(seed)
        self.L, self.W, self.H = container_size
        self.height_map = np.zeros((self.L, self.W), dtype=np.int16)

        if change_items:
            self.items = []
            self.num_items, items, _ = generate_bpp_data(num_items, num_items_min, num_items_max, container_size)
            for item in items:
                self.items.append(Item(item))
            if self.num_items == 0:
                raise ValueError("No items to place. Please ensure num_items is greater than 0.")
        
            self.action_space = spaces.Discrete(self.num_items)
            self.available_mask = np.ones(self.num_items, dtype=bool)
    
        self.placed_items = []
        self.current_step = 0
        self.fail_time = 0
        if not change_items:
            self.available_mask = np.ones(self.num_items, dtype=bool)
    
        return self.get_state_representation()

    def step(self, action, stop):
        assert self.action_space.contains(action), f"Invalid action {action}"
        if not self.available_mask[action]:
            raise ValueError(f"Action {action} corresponds to an item that has already been placed.")

        item = self.items[action]
        success = self.placement_policy(item)
        done = np.all(~self.available_mask) or self.fail_time >= self.num_items * 15 or stop
        reward = 0
 
        if success:
            height_reward = 1 - (item.position.z / self.H) 
            volume_reward = item.vol / (self.L * self.W * self.H)
            reward = (volume_reward + 5 * height_reward)
            self.fail_time = 0
        else:
            self.fail_time += 1
        
        if done:
            final_reward = self.compute_utilization()
            utilization_bonus = 150 * final_reward
            reward += utilization_bonus
            self.fail_time = 0
        
        return self.get_state_representation(), reward, done, {}

    def find_fit_location(self, block_size):
        """允许底面有高度差异的放置位置查找"""
        width, depth, height = block_size
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

    def placement_policy(self, item):
        for rotation in range(6):  # All rotations
            item.set_rotation(rotation)
            location = self.find_fit_location(item.current_state)
            if location:
                x, y, z = location
                width, depth, height = item.current_state
                if x + width <= self.L and y + depth <= self.W and z + height <= self.H:
                    if self.check_collision(x, y, z, width, depth, height):
                        continue

                    if self.check_suspension(x, y, width, depth, z):
                        continue
                
                    p = Position([x, y, z])
                    item.set_position(p)
                    item.set_placed(True)
                    self.placed_items.append(item)
                    self.update_height_map(item)
                    return True
        return False

    def check_collision(self, x, y, z, width, depth, height):
        for placed_item in self.placed_items:
            px, py, pz = placed_item.position.position
            pw, pd, ph = placed_item.current_state
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

    def update_height_map(self, item):
        width, depth, height = item.current_state
        x, y, z = item.position.position
        self.height_map[x:x+width, y:y+depth] = np.maximum(
            self.height_map[x:x+width, y:y+depth],
            z + height
        )

    def compute_utilization(self):
        total_placed_volume = sum(item.vol for item in self.placed_items)
        total_container_volume = self.H * self.L * self.W
        reward = total_placed_volume / total_container_volume
        return reward
    
    def get_state_representation(self):
        height_map_normalized = self.height_map / self.H
        available_mask = self.available_mask.astype(float)
        remaining_items = []
        for i, item in enumerate(self.items):
            if self.available_mask[i]:
                remaining_items.append([
                    item.x / self.L, 
                    item.y / self.W, 
                    item.z / self.H
                ])

        avg_height = np.mean(self.height_map) / self.H
        max_height = np.max(self.height_map) / self.H
        height_variance = np.var(self.height_map) / (self.H * self.H)
    
        state_representation = {
            "height_map": torch.tensor(height_map_normalized, dtype=torch.float32),
            "mask": torch.tensor(available_mask, dtype=torch.float32),
            "remaining_items": torch.tensor(remaining_items, dtype=torch.float32),
            "statistics": torch.tensor([avg_height, max_height, height_variance], dtype=torch.float32)
        }
        return state_representation

    def render(self, mode='print'):
        if mode == 'print':
            """
            print("Height Map:")
            print(self.height_map)
            """  
        elif mode == 'human':
            plt.figure()
            plt.imshow(self.height_map, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Height Map')
            plt.show()
            plt.savefig("1.png")

    def close(self):
        plt.close()


        
class Encoder(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.deconv1(x)))
        x = F.relu(self.batch_norm2(self.deconv2(x)))
        x = self.deconv3(x)
        return x

class ActorNetwork(nn.Module):
    def __init__(self, num_items_max):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        cnn_output_size = 128 * (100 // (2 ** 3)) * (100 // (2 ** 3))
        
        self.fc1 = nn.Linear(cnn_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_policy = nn.Linear(256, num_items_max)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, state, available_mask):
        height_map = state["height_map"].unsqueeze(0).unsqueeze(0)
        x = self.encoder(height_map)
        
        spatial_features = self.decoder(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.fc_policy(x)

        num_items = int(available_mask.size(0))
        masked_logits = logits[:, :num_items]
        masked_logits = masked_logits.masked_fill(~available_mask.bool(), float('-inf'))
        action_probs = F.softmax(masked_logits, dim=-1)
        
        return action_probs

class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        
        cnn_output_size = 128 * (100 // (2 ** 3)) * (100 // (2 ** 3))
        
        self.fc1 = nn.Linear(cnn_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_value = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, state):
        height_map = state["height_map"].unsqueeze(0).unsqueeze(0)
        x = self.encoder(height_map)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        
        return value

class PPOAgent:
    def __init__(self, num_items_max, learning_rate=1e-4, model_path=None):
        self.actor = ActorNetwork(num_items_max)
        self.critic = CriticNetwork()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            print(f"Loaded model from {model_path}")

    def select_action(self, state, available_mask):
        with torch.no_grad():
            action_probs = self.actor(state, available_mask)
            distribution = Categorical(action_probs)
            action = distribution.sample()
            while action.item() >= state["mask"].size(0):
                action = distribution.sample()
            log_prob = distribution.log_prob(action)
            value = self.critic(state)
        return action.item(), log_prob, value

    def update(self, states, actions, log_probs, rewards, values, available_masks, next_states):
        actions = torch.tensor(actions, device=self.device)
        old_log_probs = torch.stack(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        old_values = torch.stack(values).squeeze()
    
        with torch.no_grad():
            next_values = torch.stack([self.critic(state) for state in next_states]).squeeze()
            advantages = rewards + 0.99 * next_values - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
        for _ in range(4):
        # Process each state individually
            new_log_probs_list = []
            value_list = []
            entropy_list = []
        
            for state, action, mask in zip(states, actions, available_masks):
                action_prob = self.actor(state, mask)
                distribution = Categorical(action_prob)
                new_log_prob = distribution.log_prob(action)
                new_log_probs_list.append(new_log_prob)
                entropy_list.append(distribution.entropy())
            
                value = self.critic(state)
                value_list.append(value)
        
            new_log_probs = torch.stack(new_log_probs_list)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        
            policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            entropy_loss = -0.01 * torch.stack(entropy_list).mean()
        
            new_values = torch.stack(value_list).squeeze()
            value_loss = F.mse_loss(new_values, rewards)
        
            total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
        return total_loss.item()


def save_model(agent, path):
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
    }, path)
