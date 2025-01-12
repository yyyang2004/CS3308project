from generate import *
from BPP import *

def evaluate_model(model_path, num_test_episodes=100):

    env = BoxEnv(num_items=0, container_size=CONTAINER_SIZE, 
                 num_items_min=NUM_ITEM_MIN, num_items_max=NUM_ITEM_MAX)
    agent = PPOAgent(NUM_ITEM_MAX, model_path=model_path)
    
    utilization_rates = []
    episode_rewards = []
    num_items_placed = []
    max_heights = []
    
    for episode in tqdm(range(num_test_episodes)):
        state = env.reset(change_items=True, num_items_min=35, num_items_max=50)
        episode_reward = 0
        items_placed = 0
        
        while True:
            available_mask = torch.tensor(env.available_mask, device=agent.device)
            
            state = {
                "height_map": state["height_map"].to(agent.device),
                "mask": state["mask"].to(agent.device)
            }
            
            action, _, _ = agent.select_action(state, available_mask)
            
            if action >= env.num_items:
                continue
                
            try:
                next_state, reward, done, _ = env.step(action, False)
                if reward > 0:
                    items_placed += 1
                episode_reward += reward
                
                if done:
                    break
                    
                state = next_state
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                break
        
        utilization_rate = env.compute_utilization()
        max_height = np.max(env.height_map)
        
        utilization_rates.append(utilization_rate)
        episode_rewards.append(episode_reward)
        num_items_placed.append(items_placed)
        max_heights.append(max_height)
        
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1} Results:")
            print(f"Utilization Rate: {utilization_rate:.3f}")
            print(f"Items Placed: {items_placed}")
            print(f"Max Height: {max_height}")
            print(f"Episode Reward: {episode_reward:.3f}")
    
    avg_utilization = np.mean(utilization_rates)
    std_utilization = np.std(utilization_rates)
    avg_items_placed = np.mean(num_items_placed)
    avg_max_height = np.mean(max_heights)
    avg_reward = np.mean(episode_rewards)
    
    print("\nOverall Test Results:")
    print(f"Average Utilization Rate: {avg_utilization:.3f} ± {std_utilization:.3f}")
    print(f"Average Items Placed: {avg_items_placed:.2f}")
    print(f"Average Max Height: {avg_max_height:.2f}")
    print(f"Average Episode Reward: {avg_reward:.2f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(utilization_rates, bins=20)
    plt.title('Utilization Rate Distribution')
    plt.xlabel('Utilization Rate')
    plt.ylabel('Count')
    
    plt.subplot(132)
    plt.hist(num_items_placed, bins=20)
    plt.title('Items Placed Distribution')
    plt.xlabel('Number of Items')
    plt.ylabel('Count')
    
    plt.subplot(133)
    plt.hist(max_heights, bins=20)
    plt.title('Max Height Distribution')
    plt.xlabel('Height')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    return {
        'utilization_rates': utilization_rates,
        'num_items_placed': num_items_placed,
        'max_heights': max_heights,
        'episode_rewards': episode_rewards,
        'avg_utilization': avg_utilization,
        'std_utilization': std_utilization
    }

if __name__ == "__main__":
    model_path = 'best_policy.pth'
    results = evaluate_model(model_path, num_test_episodes=50)
