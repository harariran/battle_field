from .controller import Controller
from agents.team import *


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
