# Gymnasium

It has four main functions: `make()`, `Env.reset()`, `Env.step()` and `Env.render()`

At the core, Gym is a high level python class repping a markov decision proces
A markov decision is an assumption that the current state only depends on a finite/fixed number of previous states.
Thus each subsequent event occurs based on the probability of the event before it.
Hidden markov models: there is a hidden/unknown state that affects the occurence of an event. We try to infer the probabilities of these states.from our observations.(observed states)

It is used for single agent rl environments.
It gives the ability to generate an initial state, transition given an action and to visualise  the environment.


## Initialising envs

`env = gym.make("env name")`
The `make()` function initialises and returns an env.
To see all envs that are creatable do `env = gym.pprint_registry()`
Available env's:
===== classic_control =====
Acrobot-v1             CartPole-v0            CartPole-v1
MountainCar-v0         MountainCarContinuous-v0 Pendulum-v1
===== phys2d =====
phys2d/CartPole-v0     phys2d/CartPole-v1     phys2d/Pendulum-v0
===== box2d =====
BipedalWalker-v3       BipedalWalkerHardcore-v3 CarRacing-v3
LunarLander-v3         LunarLanderContinuous-v3
===== toy_text =====
Blackjack-v1           CliffWalking-v0        FrozenLake-v1
FrozenLake8x8-v1       Taxi-v3
===== tabular =====
tabular/Blackjack-v0   tabular/CliffWalking-v0
===== mujoco =====
Ant-v2                 Ant-v3                 Ant-v4
Ant-v5                 HalfCheetah-v2         HalfCheetah-v3
HalfCheetah-v4         HalfCheetah-v5         Hopper-v2
Hopper-v3              Hopper-v4              Hopper-v5
Humanoid-v2            Humanoid-v3            Humanoid-v4
Humanoid-v5            HumanoidStandup-v2     HumanoidStandup-v4
HumanoidStandup-v5     InvertedDoublePendulum-v2 InvertedDoublePendulum-v4
InvertedDoublePendulum-v5 InvertedPendulum-v2    InvertedPendulum-v4
InvertedPendulum-v5    Pusher-v2              Pusher-v4
Pusher-v5              Reacher-v2             Reacher-v4
Reacher-v5             Swimmer-v2             Swimmer-v3
Swimmer-v4             Swimmer-v5             Walker2d-v2
Walker2d-v3            Walker2d-v4            Walker2d-v5
===== None =====
GymV21Environment-v0   GymV26Environment-v0

## Env interaction(`Env.`)

The agent environment loop is as follows.
$Agent(policy) --action--> environment --reward--observation--> agent$

[code](https://github.com/toxxicblood/learning/blob/main/AI/quick_learn/openAI_gym/gymnasium_practice.py)

There are different actions :
`Env.render()`
`Env.reset()`
`Env.step()`
Environment ending is called a terminal state
In gym if the env has terminated, it is returned by `step()`  as the third variable `terminated`
If we want the env to end after a certain number of steps, here the env issues a `truncated` signal.
If either `terminated` or `truncated` is true we end the episode.
If we wish to restart the env tho, we  use `env.reset()`

## Action and observation spaces

Every env specifies formant of valid actions and observations with the `action_space` and `observation_space` attributes.
This helps know the expected inputs and outpusts of the env as all valid acions and observations should be contained within their respective spaces.

More importantly, `env.action_space` and `env.observation_space` are both instances of `Space` a high level clas that provides key functions : `space.contains() and space .sample()

Gym has support for a wide range of spaces that users might need.

## Modifying the env

- Wrappers are used to mod an env wihout having to alter the underlying code
- To wrap an env first initialise base env then pass env along with parameters to the wrapper's constructor
TimeLimit: Issues a truncated signal if a maximum number of timesteps has been exceeded (or the base environment has issued a truncated signal).

ClipAction: Clips any action passed to step such that it lies in the base environment’s action space.

RescaleAction: Applies an affine transformation to the action to linearly scale for a new low and high bound on the environment.

TimeAwareObservation: Add information about the index of timestep to observation. In some cases helpful to ensure that transitions are Markov.


## Training an env

it is not good to call `env.render()` in the training loop because it slows down training by a lot. Tather try to build an extra loop to eval and showcase agent after training

- To train an agent we complete one teration called an episode