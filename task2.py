import copy
import torch
import numpy as np
from task1.BPP import *

# 假设已经有了任务一中定义的以下类和函数
# BoxEnv, ActorNetwork, CriticNetwork, PPOAgent, save_model, train

L = 100
W = 100
H = 100

# 定义深度优先搜索函数与策略网络结合
def dfs_search(env, state, agent, depth=0, max_depth=10, path=None):
    if path is None:
        path = []

    if depth >= max_depth:
        print(f"Reached maximum depth {max_depth}")
        return None

    available_mask = state["mask"].to(agent.device)
    action_probs = agent.actor(state, available_mask)
    _, sorted_actions = torch.sort(action_probs, descending=True)

    for action in sorted_actions[0].tolist():
        if not available_mask[action]:
            continue

        # 深拷贝环境以保存当前状态
        original_env = copy.deepcopy(env)

        try:
            print(f"At depth {depth}, trying action {action}")
            next_state, reward, done, _ = env.step(action, False)
            next_state = {
                "height_map": next_state["height_map"].to(agent.device),
                "mask": next_state["mask"].to(agent.device)
            }

            path.append((action, reward))

            if done:
                print(f"Search completed at depth {depth} with reward {reward}")
                return path
            else:
                result = dfs_search(env, next_state, agent, depth + 1, max_depth, path)
                if result is not None:
                    return result
        except Exception as e:
            print(f"Exception occurred: {e}")

        # 回溯到之前的状态
        env = original_env
        path.pop()
        print(f"Backtracking at depth {depth}")

    return None

# 测试函数
def test_combined_algorithm():
    env = BoxEnv(num_items=0, num_items_min=10, num_items_max=30)
    agent = PPOAgent(NUM_ITEM_MAX, model_path='best_policy.pth')

    state = env.reset()
    result = dfs_search(env, state, agent)
    if result is not None:
        total_reward = sum([reward for _, reward in result])
        print(f"Total reward: {total_reward}")
        print("Action path:", [action for action, _ in result])
    else:
        print("Search failed.")

if __name__ == "__main__":
    test_combined_algorithm()
