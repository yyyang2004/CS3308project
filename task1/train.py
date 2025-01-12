from generate import *
from BPP import *

def train(num_episodes=500, max_steps=600, model_path=None):
    writer = SummaryWriter(log_dir='./logs')
    env = BoxEnv(num_items=0, container_size=CONTAINER_SIZE, num_items_min=NUM_ITEM_MIN, num_items_max=NUM_ITEM_MAX)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_file_dir, model_path) if model_path else os.path.join(current_file_dir, 'best_policy.pth')

    agent = PPOAgent(NUM_ITEM_MAX, model_path=model_path)
    
    batch_size = 8
    update_interval = 128
    mean_utilization_rate = 0
    
    max_reward = float('-inf')
    max_utilization_rate = float('-inf')
    episode_rewards = []
    mean_utilization_rates = []
    
    states_batch = []
    next_states_batch = []
    actions_batch = []
    log_probs_batch = []
    values_batch = []
    available_masks_batch = []
    rewards_batch = []
    
    for episode in tqdm(range(num_episodes)):
        if episode <= 200:
            state = env.reset(change_items=(episode % 10 == 0), num_items_min=30, num_items_max=50)
        elif episode <= 400:
            state = env.reset(change_items=(episode % 8 == 0), num_items_min=30, num_items_max=50)
        else:
            state = env.reset(change_items=(episode % 5 == 0), num_items_min=35, num_items_max=50)

        episode_reward = 0
        
        
        for step in range(max_steps):
            available_mask = torch.tensor(env.available_mask, device=agent.device)
            stop = (step == max_steps - 1)

            state = {
                "height_map": state["height_map"].to(agent.device),
                "mask": state["mask"].to(agent.device)
            }

            action, log_prob, value = agent.select_action(state, available_mask)
            
            if action >= env.num_items:
                continue
                
            try:
                next_state, reward, done, _ = env.step(action, stop)
            except AssertionError as e:
                print(f"Error taking step: {e}")
                break

            states_batch.append(state)
            next_states_batch.append({
                "height_map": next_state["height_map"].to(agent.device),
                "mask": next_state["mask"].to(agent.device)
            })
            actions_batch.append(action)
            log_probs_batch.append(log_prob)
            values_batch.append(value)
            available_masks_batch.append(available_mask)
            rewards_batch.append(reward)

            if len(states_batch) >= update_interval:
                for idx in range(0, len(states_batch), batch_size):
                    end_idx = min(idx + batch_size, len(states_batch))
                    batch_slice = slice(idx, end_idx)
                    
                    loss = agent.update(
                        states_batch[batch_slice],
                        actions_batch[batch_slice],
                        log_probs_batch[batch_slice],
                        rewards_batch[batch_slice],
                        values_batch[batch_slice],
                        available_masks_batch[batch_slice],
                        next_states_batch[batch_slice]
                    )
                    
                    writer.add_scalar('Loss', loss, (episode + 1) * (step + 1))

                states_batch.clear()
                next_states_batch.clear()
                actions_batch.clear()
                log_probs_batch.clear()
                values_batch.clear()
                available_masks_batch.clear()
                rewards_batch.clear()

                torch.cuda.empty_cache()
            
            state = next_state
            episode_reward += reward
            
            if done or step >= max_steps - 1:
                break
            
        utilization_rate = env.compute_utilization()
        episode_rewards.append(episode_reward)

        if episode_reward > max_reward:
            max_reward = episode_reward
        
        if utilization_rate > max_utilization_rate:
            max_utilization_rate = utilization_rate
            if episode >= 100:
                save_model(agent, os.path.join(current_file_dir, 'best_policy.pth'))
                print(f"New best model saved with utilization rate: {utilization_rate:.3f}")
        
        print(f'Utilization Rate: {utilization_rate:.3f}')
        writer.add_scalar('Utilization Rate', utilization_rate, episode + 1)
        mean_utilization_rate += utilization_rate

        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            mean_utilization_rate /= 10
            
            print(f'Episode {episode+1}:')
            print(f'Average Reward: {avg_reward:.3f}')
            print(f'Max Reward: {max_reward:.3f}')
            print(f'Mean Utilization Rate: {mean_utilization_rate:.3f}')
            print(f'Max Utilization Rate: {max_utilization_rate:.3f}')
            
            env.render(mode='print')
            writer.add_scalar('Average Reward', avg_reward, episode + 1)
            writer.add_scalar('Mean Utilization Rate', mean_utilization_rate, episode + 1)
            
            mean_utilization_rates.append(mean_utilization_rate)
            
            if mean_utilization_rates[-1] >= max(mean_utilization_rates) and episode >= 100:
                save_model(agent, os.path.join(current_file_dir, 'best_policy.pth'))
                print(f"New best model saved with mean utilization rate: {mean_utilization_rates[-1]:.3f}")
                
            mean_utilization_rate = 0
        
        if len(mean_utilization_rates) > 2:
            if abs(mean_utilization_rates[-1] - mean_utilization_rates[-2]) < 1e-4:
                print(f"Early stopping at episode {episode + 1}")
                break
    
    writer.close()
    return agent

if __name__ == "__main__":
    best_model_path = 'best_policy.pth'
    agent = train(model_path=best_model_path)
