"""
Lets start with our imports
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from warnings import filterwarnings

from gym.spaces import Discrete
from pettingzoo.magent import battlefield_v5, battle_v4

from DMs.simple_planner import Simple_DM
from agents import Agent
from environments import EnvWrapperPZ
from utils.functions import CreateDecentralizedAgents
from DMs.PPO_HL1 import PPO_Low_level_DecisionMaker
from DMs.simple_DMs import *
from DMs.simple_planner import Simple_DM
from DMs.simple_planner2 import Simple_DM2
from agents.team import Team
from coordinators import coordinator
from utils import constants as const, performance, factory
from agents.agent import DecisionMaker
from coordinators.coordinator import coordinator
from control import Controller

"""
we use MAC (updated), and ai_dm from sarah's git tools
"""

# A Wrapper for the pettingzoo environment within MAC
class BattleFieldEnv(EnvWrapperPZ):
    def __init__(self, env):
        super().__init__(env)
        self.obs = self.env.reset()
        # print(self.obs)

    def step(self, joint_action):
        return self.env.step(joint_action)

    def observation_to_dict(self, obs):
        return obs

    def final_result(self):
        return self.env.team_sizes

    def reset(self):
        return self.env.reset()



# we will use a function that creates and reset our environment - we can set our rewards as necessary
def CreateEnvironment_Battle(minimap=False):
    # Create and reset PettingZoo environment
    BF_env = battle_v4.parallel_env(map_size=45, minimap_mode=minimap, step_reward=-0.005, dead_penalty=-7,
                                         attack_penalty=-0.01, attack_opponent_reward=0.8, max_cycles=400,
                                         extra_features=False)
    BF_env.reset()

    # Create a MAC from the PZ environment
    return BattleFieldEnv(BF_env)


"""
we define a new class - Team that gets a decision maker and coordinator - and control the team accordingly
"""
class Team:
    def __init__(self, team_name, agents , coordinator : coordinator =None, plan_lenght = 1):
        self.team_name = team_name
        self.agents = agents
        try:
            self.agents_names = agents.keys()
        except:
            self.agents_names = None
        self.coordinator = coordinator
        self.plan_lenght = plan_lenght

    def get_agents(self):
        return self.decision_makers

    def get_agents_names(self):
        return self.agents_names

    def get_coordinator(self):
        return self.coordinator

    def get_joint_action(self, observation):
        """Returns the joint action

                Args:
                    observation (dict): the current observatins

                Returns:
                    dict: the actions for the agents
                """
        team_observations = {agent_id: observation[agent_id]
                            for agent_id in self.agents_names if agent_id in observation.keys() and agent_id}
        joint_action = {}
        joint_plan = {}
        for agent_name in team_observations.keys():
            if self.coordinator is not None:  # If there's a coordinator, the decision maker returns a plan
                try:
                    plan = self.agents[agent_name].get_decision_maker().get_plan(team_observations[agent_name],
                                                                                 self.plan_length)
                except:
                    plan = self.agents[agent_name].get_decision_maker().get_action(team_observations[agent_name])
                if isinstance(plan, int):  # correct single action plan to a list
                    plan = [plan]
                joint_plan[agent_name] = plan
            else:
                action = self.agents[agent_name].get_decision_maker().get_action(team_observations[agent_name])
                joint_action[agent_name] = action

        # The coordinator's approve_joint_action returns the next joint action, after considering the joint plan
        if self.coordinator is not None:
            joint_action = self.coordinator.approve_joint_plan(joint_plan)
        return joint_action

"""
now we use MAC controller to build our Teams controller
"""
class TeamsController(Controller):
    """controller for running teams in the game
        each can have its own coordinator/agents
            """

    def __init__(self, env, teams):
        # initialize super class
        super().__init__(env)
        self.teams = teams


    def get_joint_action(self, observation):
        """Returns the joint action

        Args:
            observation (dict): the current observatins

        Returns:
            dict: the actions for the agents
        """

        joint_action = {}
        for team in self.teams:
            joint_action.update(team.get_joint_action(observation))
        return joint_action


"""
 function - Creates identical agents with the same decision maker for agents names and optional coordinator for central 'control'
"""
def CreateDecentralizedAgentsTeam(env, team_name, decision_maker,agent_names,coordinator=None):
    if isinstance(decision_maker,DecisionMaker):
        decentralized_agents = {
            agent_id: Agent(copy.deepcopy(decision_maker))
            for agent_id in agent_names
        }
    else:
        decentralized_agents = {
            agent_id: Agent(decision_maker(env.action_spaces[agent_id]))
            for agent_id in agent_names
        }
    return Team(team_name,decentralized_agents,coordinator)

# Create and run rounds using teams -> ruturns results and average of all rounds
def CreateTeamsController(env, teams, render=True,max_iters=1000, rounds=1):
    teams_controller = TeamsController(env, teams)

    average_score_and_rewards = [["red_team_avarage",0,0],["blue_team_average",0,0]]
    results = []
    for i in range(rounds):
        # Running the decentralized agents
        teams_controller.run(render=render, max_iteration=max_iters)

        # save and return results of this round
        total_rewards = teams_controller.total_rewards
        red_total_rewards = sum(total_rewards[item] for item in total_rewards if "red" in item)
        blue_total_rewards = sum(total_rewards[item] for item in total_rewards if "blue" in item)
        last_lives = env.final_result()
        result = [["red_team",last_lives[0],red_total_rewards],["blue_team",last_lives[1],blue_total_rewards]]
        results.append(result)
        for m in range(1,3):
            average_score_and_rewards[0][m]+=(result[0][m]-average_score_and_rewards[0][m])/(i+1)
            average_score_and_rewards[1][m]+=(result[1][m] - average_score_and_rewards[1][m]) / (i + 1)

    # print(f"{result}")
    return results, average_score_and_rewards

"""
now after we set the right architecture and helping functios - we are ready to try our controller 'brains'
"""

"""
first we will make a simple random decition maker for a 'stupid' baseline
"""

# random DM:
from agents import DecisionMaker

class RandomDecisionMaker(DecisionMaker):
    def __init__(self, action_space):
        self.space = action_space

    def get_action(self, observation):

        #TODO this assumes that the action space is a gym space with the `sample` funcition
        if isinstance(self.space, dict):
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()

    # Random plan - the result of a random sequence of get_action calls
    def get_plan(self, observation, plan_length):
        plan = [self.get_action(None) for _ in range(0, plan_length)]  # Blind action choice
        return plan

"""
We will also make a simple DM that always stand still (agents use only stay action) 
"""

class Stay_DM(DecisionMaker):
    def __init__(self, action_space, action=6):
        self.action_space=action_space
        self.action = action

    def get_action(self, observation):
        return self.action

"""
Round 1 - stay DM VS random DM
"""
# test our teams - blue-random VS red-stay

# first - set the env and agents names
env = CreateEnvironment_Battle(minimap=False)
agents = env.get_env_agents()
red_agents  = [agent for agent in agents if "red" in agent]
blue_agents = [agent for agent in agents if "blue" in agent]

# set our teams:
team1 = CreateDecentralizedAgentsTeam(env,"blues",Stay_DM,blue_agents)
team2 = CreateDecentralizedAgentsTeam(env,"reds",Stay_DM,red_agents)
# run 1 round
CreateTeamsController(env,[team1,team2],max_iters=100,rounds=2)




class BattleFieldSingleEnv():
    def __init__(self, env,blue_other_dm, red_DM, agent="blue_0"):
        self.env = env
        self.action_space = env.action_spaces[agent]
        self.observation_space = env.observation_spaces[agent]
        self.agent_name = agent
        self.agents = CreateDecentralizedAgents(env, blue_other_dm, red_DM)
        self.others_blue_DM = blue_other_dm
        self.others_red_DM = red_DM
        self.all_obs = self.env.reset()
        self.obs = self.all_obs[agent]
        self.metadata = None
        # print(self.obs)


    def step(self, action):
        joint_action = {}
        for agent_name in self.all_obs.keys():
            act = self.agents[agent_name].get_decision_maker().get_action(self.all_obs[agent_name])
            joint_action[agent_name] = act
        joint_action[self.agent_name] = action
        self.all_obs, reward, done, self.metadata = self.env.step(joint_action)
        try:
            self.obs = self.all_obs[self.agent_name]
        except:
            return (None, None, True, self.metadata)
        return (self.obs, reward[self.agent_name], done[self.agent_name], self.metadata)

    def observation_to_dict(self, obs):
        return obs

    def render(self):
        return self.env.render()

    def reset(self):
        self.all_obs = self.env.reset()
        self.obs = self.all_obs[self.agent_name]
        return self.obs

    def print_env_info(self):
        for i, agent in enumerate(self.env.agents, 1):
            print(f'- agent {i}: {agent}')
            print(f'\t- observation space: {self.env.observation_space(agent)}')
            print(f'\t- action space: {self.env.action_space(agent)}')
            print(f'\t- action space sample: {self.env.action_space(agent).sample()}')