import gym
import torch

from torch.nn import DataParallel
from agent import Agent

EPISODES = 2_001

env = gym.make("LunarLander-v2")
agent = Agent(
    in_size=len(env.observation_space.low),
    out_size=env.action_space.n,
    epsilone=0.1,
    min_eps=0.1,
    eps_decay=0.1,
    learning_rate=0.01,
    batch_size=64,
    gamma=0.99
)

# agent.network = DataParallel(agent.network)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent.load_model("./lunar_models/lunar-2Q_model_1200.torch", map_location=DEVICE)
agent.network.to(DEVICE)

episode_rewards = []
for episode in range(EPISODES):
    step = 0
    new_state = env.reset()
    done = False

    for i in range(500):
        if episode % 50 == 0:
            env.render()

        state = new_state
        action = agent.pick_action(torch.tensor(new_state).unsqueeze(0).float())
        new_state, reward, done, _ = env.step(action)

        if done:
            print(f"episode: {episode} reward: {sum(episode_rewards):.2f}, iteration {i}")
            episode_rewards = []
            agent.memory.add(state, action, reward, None)
            episode_rewards.append(reward)
            break

        agent.memory.add(state, action, reward, new_state)
        episode_rewards.append(reward)
        step += 1

    if episode % 10 == 0 and episode > 0:
        agent.memory.shuffle()
        agent.train()

    if episode % 100 == 0 and episode != 0:
        agent.save_model(f"lunar_models/lunar-2Q_model_{episode}.torch")
