from abc import ABC, abstractmethod


class Agent:

    def __init__(self, decision_maker, sensor_function=None, message_filter=None, agent_name=None):
        self.decision_maker = decision_maker
        self.sensor_function = sensor_function or (lambda x: x)  # default to identity function
        self.message_filter = message_filter
        self.agent_name = agent_name

    def get_decision_maker(self):
        return self.decision_maker

    def get_observation(self, state):
        return self.sensor_function(state)


class DecisionMaker(ABC):
    """
    An abstract class for choosing an action, part of an agent.
    (An agent can have one or several of these)
    """

    @abstractmethod
    def get_action(self, observation):
        pass

