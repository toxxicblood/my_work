#Basic game info
    #observations provide target and agent locatins
    # there are 4 discrete actins in the env (right, up, left, down)
    # the env terminates when the agent navigates to the grid where the target is located
    # the agent is rewarded when it reaches the target

#as with all envs custon=m envs inherit from gymnasium.Env
# the most important thing is defining observatin and action space#
# define set of possible inputs(actions) and outputs(observations)
#with our four inputs we will use a `Discrete(4)`` space with four options
# imagin our observation looks like this`:{"agent": array([1, 0]), "target": array([0, 3])}
#showing the agent's and target's locations


from typing import Optional
import numpy as np
import gymnasium as gym

class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5)
        #the  size of the squrar grid
        self.size = size

        # define agent and target location randomly chosen in reset and updataed in step
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        #observations are dicts with the agents and target location
        # each location is encoded as an element of {0, ..., size -1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size -1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size -1, shape=(2,), dtype=int),
            }
        )

        #wwe have 4 actions corresponding to right up left down
        self.action_space = gym.spaces.Discrete(4)
        # dictonary maps the abstract actions to directions on the grid
        self.action_to_direction = {
            0:np.array([1,0]), #right
            1:np.array([0,1]), #up
            2:np.array([-1,0]), #left
            3:np.array([0,-1]), #down
        }

    ### Constructing observations
    """since we need to compute obsertions with env.reset and env.step its  easie to have
    a method that translates the env's state into an observation"""
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    #we can also construct a function for the auziliary info returned by env.reset and env.step
    # in our case we want the manhattan distance btw agent and target
    def _get_info(self):
        return{
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    ### Reset function
    """itss purpose is to initiate a new episode for an env
    it has two params:
        the seed: used to initialise  a random number generator
        options: used to specify values used within reset

    in our env the reeset needs to randomly choose the agent's and targets locations
    (repeat if theyre in the same position
    reset(0 )returns  a tuple of (initial observatin, auziliary info) """

    def reset(self, seed: Optional[int]= None, options: Optional[dict] = None):
        #we need the following line to seesd self.np_random
        super().reset(seed=seed)

        #choose agent loccation uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)


        # we will sample targets location randomly till it doesnt coincide with the agent
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    ### Step function
    """this method contains the most logic
    It acceptsan action and computes the state of env after applying the action
    returns a tuple of :
         the next observatin,, reward, if env is terminated, if env has truncated, aux info
    In our env we want the following to happen:
        use self._action_to_direction to convert discrete action to a grid  direction
            also clip agent location to stay within bounds
        compute agent reward by checkin if agent position is equal to target location
        apply a time limit to env during make , set truncated to false
        use _get_obs and getinfo to obtain agens observation and aux info
        """

    def step(self, action):
        # map action (element of {o,1,2,3}) to direcion we walk in
        direction = self._action_to_direction[action]
        #use np.clip to make sure we dont leave grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0 ,self.size -1
        )
        # an env is complete oly if agent has reached target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated= False
        reward = 1 if terminated else 0
        observation = self._get_obs
        info = self._get_info

        return observation, reward, terminated, info


    ### Registering and making the environment
    """
    env id consists of three components, two are optional
    Optional namespace : gymnasium.env
    mandatory name : gridworld
    optional version : v-

    """
    gym.register(
        id= "gymnasium_env/Gridworld-v0",
        entry_point=GridWorldEnv,

    )
    ###Recording agents
    """during traning is may be interesting to record gent behaviour and
    record totoal reward accumlated
     this can be achieved using two wrappers:
        RecordEpisodeStatistics:tracks data such as total reward, ep length and time taken
          and RecordVideo:generates an mp4 vid of agents using the envs
            """