import copy

from agents.agent import DecisionMaker
from coordinators.coordinator import coordinator

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