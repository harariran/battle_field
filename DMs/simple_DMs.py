import copy

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

class Do_action_DM:
    def __init__(self, action_space, action=6):
        self.action_space=action_space
        self.steps = 0
        self.action = action

    def get_action(self, observation):
        self.steps+=1



        if (self.steps%3 ==0):
            return self.action
        elif (self.steps%3==1):
            return self.action_space.sample()
        else:
            return 2

class Stay_DM(DecisionMaker):
    def __init__(self, action_space, action=6):
        self.action_space=action_space
        self.steps = 0
        self.action = action

    def get_action(self, observation):
        self.steps+=1
        if (self.steps  < 9 ):
            return self.action
        else:
            return 6

class GreedyDecisionMaker(DecisionMaker):
    def __init__(self, env):
        # We create a full copy of the environment to use it for planning but without changing the original environment
        self.env = copy.deepcopy(env)

    def get_action(self, observation):

        if isinstance(self.space, dict):
            return {agent: self.space[agent].sample() for agent in self.space.keys()}
        else:
            return self.space.sample()


# A decision maker with a pre-defined joint plan, for simulation
class SimDecisionMaker(DecisionMaker):
    def __init__(self, joint_plan):
        self.joint_plan = joint_plan
        self.current_action_index = -1
        self.min_plan_length = min([len(plan) for plan in joint_plan.values()])

    def get_action(self, observation):
        self.current_action_index += 1
        if self.current_action_index >= self.min_plan_length:
            return None
        return {agent_id: plan[self.current_action_index] for (agent_id, plan) in self.joint_plan.items()}