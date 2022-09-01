from abc import ABC, abstractmethod
import time


class Controller(ABC):
    """An abstract controller class, for other controllers
    to inherit from
    """

    # init agents and their observations
    def __init__(self, env):
        self.env = env
        self.agent_ids = self.env.get_env_agents()
        self.total_rewards = {agentname: 0 for agentname in self.agent_ids}

    def run(self, render=False, max_iteration=None):
        """Runs the controller on the environment given in the init,
        with the agents given in the init

        Args:
            render (bool, optional): Whether to render while runngin. Defaults to False.
            max_iteration ([type], optional): Number of steps to run. Defaults to infinity.
        """
        done = False
        index = 0
        observation = self.env.get_env().reset()
        # self.total_rewards = {agentname:0 for agentname in observation.keys()}
        while done is not True:
            index += 1
            if max_iteration is not None and index > max_iteration:
                break

            # display environment
            if render:
                self.env.render()
                time.sleep(0.1)


            # assert observation is in dict form
            observation = self.env.observation_to_dict(observation)

            # get actions for all agents and perform
            joint_action = self.get_joint_action(observation)
            observation, reward, done, info = self.perform_joint_action(joint_action)

            # save rewards

            for key in reward.keys():
                self.total_rewards[key]+= reward[key]
            # self.total_rewards.append(reward)

            # check if all agents are done
            if all(done.values()):
                break

        if render:
            self.env.render()

    def perform_joint_action(self, joint_action):
        return self.env.get_env().step(joint_action)

    def get_rewards_dict(self):
        return self.total_rewards

    @abstractmethod
    def get_joint_action(self, observation):
        pass
