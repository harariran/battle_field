from DMs.PPO_HL1 import PPO_Low_level_DecisionMaker
from DMs.simple_DMs import *
from DMs.simple_planner import Simple_DM
from DMs.simple_planner2 import Simple_DM2
from agents.team import Team
from coordinators import coordinator
from utils import constants as const, performance, factory


# Test a centralized controller with a random decision maker
def test_centralized_controller(env):
    factory.CreateCentralizedController(env, factory.CreateRandomAgent(env))


# Test a decentralized controller with a random decision maker
def test_decentralized_controller(env):
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, RandomDecisionMaker, RandomDecisionMaker))


# Test a simulator controller (Only runs a predefined joint plan) with random decision makers
def test_sim_controller(env):
    RDM_dict = {agent_id: RandomDecisionMaker(env.action_spaces[agent_id]) for agent_id in env.get_env_agents()}
    plan_dict = {agent_id: rdm.get_plan(None, const.LONG_PLAN_LENGTH) for (agent_id, rdm) in RDM_dict.items()}
    total_rewards, observations = factory.CreateSimulationController(env, plan_dict)
    result = performance.objective(plan_dict, observations, total_rewards)
    print(result)


# Test a coordinator within a decentralized controller, using random decision makers.
# The IdentityCoordinator approves every plan as is
def test_coordinator(env):
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, RandomDecisionMaker, RandomDecisionMaker), coordinator=coordinator.IdentityCoordinator(env), plan_length=const.PLAN_LENGTH)


# Test a coordinator within a decentralized controller.
# The SimGreedyCoodinator uses a simulator controller to simulate the joint plan,
# and it uses a greedy mechanism and a binary hard constraint between every pair of plans to decide which plan to take.
# The unapproved plans become plans with a repeated default action (Do nothing)
def test_sim_coordinator(env):
    factory.CreateDecentralizedController(env, factory.CreateDecentralizedAgents(env, Simple_DM , RandomDecisionMaker), coordinator=coordinator.SimGreedyCoordinator(env), plan_length=const.PLAN_LENGTH)

def test_sim_teams(env):
    agents = env.get_env_agents()
    red_agents  = [agent for agent in agents if "red" in agent]
    blue_agents = [agent for agent in agents if "blue" in agent]
    team1 = factory.CreateDecentralizedAgentsTeam(env,"blues",Simple_DM,blue_agents, coordinator=coordinator.IdentityCoordinator(env))
    team2 = factory.CreateDecentralizedAgentsTeam(env,"reds",Stay_DM,red_agents)
    factory.CreateTeamsController(env,[team1,team2])

def test_sim_teams2(env):
    agents = env.get_env_agents()
    red_agents  = [agent for agent in agents if "red" in agent]
    blue_agents = [agent for agent in agents if "blue" in agent]
    team1 = factory.CreateDecentralizedAgentsTeam(env,"blues",Simple_DM,blue_agents, coordinator=coordinator.IdentityCoordinator(env))
    team2 = factory.CreateDecentralizedAgentsTeam(env,"reds",Simple_DM(env.action_spaces["blue_1"], red_team=True)
                                                  ,red_agents, coordinator=coordinator.IdentityCoordinator(env))
    factory.CreateTeamsController(env,[team1,team2])

def test_sim_teams3(env):
    agents = env.get_env_agents()
    red_agents  = [agent for agent in agents if "red" in agent]
    blue_agents = [agent for agent in agents if "blue" in agent]
    team1 = factory.CreateDecentralizedAgentsTeam(env,"blues",PPO_Low_level_DecisionMaker(env.action_spaces["blue_1"]),blue_agents)
    team2 = factory.CreateDecentralizedAgentsTeam(env,"reds",Simple_DM2(env.action_spaces["blue_1"],0.5,red_team=True)
                                                  ,red_agents)
    factory.CreateTeamsController(env,[team1,team2])

def test_sim_teams4(env):
    agents = env.get_env_agents()
    red_agents  = [agent for agent in agents if "red" in agent]
    blue_agents = [agent for agent in agents if "blue" in agent]
    team1 = factory.CreateDecentralizedAgentsTeam(env,"blues",PPO_Low_level_DecisionMaker(env.action_spaces["blue_1"]),blue_agents)
    team2 = factory.CreateDecentralizedAgentsTeam(env,"reds",PPO_Low_level_DecisionMaker(env.action_spaces["blue_1"],red_team=True, model_name="HL_PPO2")
                                                  ,red_agents)
    factory.CreateTeamsController(env,[team1,team2])

def test_sim_teams5(env):
    agents = env.get_env_agents()
    red_agents  = [agent for agent in agents if "red" in agent]
    blue_agents = [agent for agent in agents if "blue" in agent]
    team1 = factory.CreateDecentralizedAgentsTeam(env,"blues",PPO_Low_level_DecisionMaker(env.action_spaces["blue_1"]),blue_agents)
    team2 = factory.CreateDecentralizedAgentsTeam(env,"reds",Simple_DM(env.action_spaces["blue_1"], red_team=True)
                                                  ,red_agents)
    result = factory.CreateTeamsController(env,[team1,team2],max_iters=50,rounds=2)
    print(result)