import gymnasium as gym

def main():
    #this is where i call an environment
    # # Create

# Initialise the environment
    env = single_episode()

def  begginer_explanation():
    env = gym.make("LunarLander-v3", render_mode="human")

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

def single_episode():
    # in this instead of being stuck in a loop, we only go for one epiisode
    env = gym.make("LunarLander-v3", render_mode="human")
    # render mode specifies how env should be visualised
    observation, info = env.reset()#we use thi to generate the first observation
    # this is where you would insert your policy

    episode_over = False
    # we define this as a variabl becausse we want to continue the loop untill the environment ends
    #this is an unknown number of tiesteps.
    while not episode_over:
        action = env.action_space.sample()#agent policy that uses the observation sand info
        observation, reward, terminated, truncated, info = env.step(action)
        #env.step(action) is where the agent does an action in the env.

        episode_over = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()