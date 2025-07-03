import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()#gets the first observation and additional info

episode_over = False
while not episode_over:
    action = env.action_space.sample()#agent policy
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
#this returns an env for users to interact with
