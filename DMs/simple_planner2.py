import copy
import random

from agents import DecisionMaker
# __author__ = 'sarah'
#
# from AI_agents.Search.problem import Problem
# import AI_agents.Search.utils as utils
# from AI_agents.Search.best_first_search import a_star

#
# class Battle_Problem(Problem):
#
#     """Problem superclass
#        supporting COMPLETE
#     """
#     def __init__(self, env, init_state, constraints=[], goal="attack"):
#         super().__init__(init_state, constraints)
#         self.goal = goal
#         self.env = env
#         self.counter = 0
#
#     # get the actions that can be applied at the current node
#     def get_applicable_actions(self, node):
#         action_list = self.env.P[node.state.get_key()].keys()
#         return action_list
#
#     # get (all) succesor states of an action and their
#     def get_successors(self, action, node):
#
#         #action_list = self.env.P[node.state.__repr__()]
#         successor_nodes = []
#         transitions = self.env.P[node.state.__str__()][action]
#         action_cost = self.get_action_cost(action, node.state)
#         for prob, next_state_key, reward, done in transitions:
#             info={}
#             info['prob'] = prob
#             info['reward'] = reward
#             next_state = utils.State(next_state_key, done)
#             successor_node = utils.Node (state=next_state, parent=node, action=action, path_cost=node.path_cost + action_cost, info=info)
#             successor_nodes.append(successor_node)
#
#         return successor_nodes
#
#     def get_action_cost(self, action, state):
#         return 1
#
#     def is_goal_state(self, state):
#         if state.is_terminal:
#             return True
#         else:
#             return False
#
#     def apply_action(self, action):
#         state, reward, done, info = self.env.step(int(action))
#         if self.goal == "attack":
#             if reward > 0 : state.is_terminal = True
#         return [state, reward, done, info]
#







class Simple_DM2(DecisionMaker):
    def __init__(self, action_space, health_th=0.5 , red_team=False):
        self.space = action_space
        self.search_counter = 10
        self.search_orientation = random.choice(["east","west","north","south"])
        self.healt = None
        self.healt_th = health_th
        self.walls = []
        self.my_team = []
        self.op_team = []
        self.is_red_team = red_team

    def set_state(self, obs):
        self.walls = []
        self.my_team = []
        self.op_team = []
        self.healt = obs[6,6,2]
        for y in range (13):
            for x in range (13):
                if x==6 and y==6:
                    continue
                if obs[y,x,0]==1:
                    self.walls.append((x-6,y-6))
                elif obs[y,x, 1] == 1:
                    self.my_team.append(((x-6,y-6),obs[y,x,2]))
                elif obs[y,x,3] == 1:
                    self.op_team.append(((x-6,y-6), obs[y,x, 4]))

    def defensive_move(self):
        if len(self.op_team)==0:
            return 6
        else:
            x = self.find_closest()
            if x[0]<0:
                return random.choice([3,8,11])
            elif x[0]>0 :
                return random.choice([1,4,9])
            if x[0]<0 :
                return random.choice([12,11,9])
            else:
                return random.choice([0,1,3])

    def search_opponent(self):
        self.search_counter-= 1
        if self.search_counter==0:
            self.search_orientation = random.choice(["east", "west", "north", "south"])
            self.search_counter=10
        if self.search_orientation=="east":
            return random.choice([3, 8, 7, 11])
        elif self.search_orientation=="west":
            return random.choice([1,4,5,9])
        elif self.search_orientation=="north":
            return random.choice([0,1,2,3])
        else:
            return random.choice([9,10,11,12])

    def attack_range(self):
        if len(self.op_team)==0 : return None
        for (x,y) in self.op_team:
            if abs(x[0])<2 and abs(x[1])<2 :
                return (x)
        return None

    def attack(self,pos):
        "translate position to related attack action"
        i = 4 + pos[0] + (3*pos[1])
        if i>4 : i-=1
        return (13 + i)

    def go_to(self,pos):
        "translate position to related  action"
        i = 6 + pos[0] + (4*pos[1])
        return i

    def find_closest(self):
        min = 13
        for x,_ in self.op_team:
            dist = abs(x[0])+ abs(x[1])
            if dist<min :
                close_pos = x
                min = dist
        return close_pos


    def check_wall(self,pos):
        if (pos in self.walls):
            return True
        else: return False

    def check_my_team(self,pos):
        if (pos in [x[0] for x in self.walls]):
            return True
        else: return False

    def act_to_pos(self,act):
        pos = [(0,-2),(-1,-1),(0,-1),(1,-1),
               (-2,0),(-1,0),(0,0),(1,0),(2,0),
               (-1,1),(0,1),(1,1),(0,2)]
        return pos[act]

    def chase_closest(self):
        close = self.find_closest()
        if close[1]>2:
            if self.check_my_team(self.act_to_pos(12)):
                return 10
            else: return 12
        if close[1]<-2:
            if self.check_my_team(self.act_to_pos(0)):
                return 2
            return 0
        if close[0]<-2:
            if self.check_wall((-2,0)) or self.check_wall((-1,0)):
                if self.check_my_team(self.act_to_pos(0)):
                    return 2
                return 0
            else: return 4
        if close[0]>2:
            if self.check_wall((2,0)) or self.check_wall((1,0)):
                if self.check_my_team(self.act_to_pos(12)):
                    return 10
                return 12
            else:
                if self.check_my_team(self.act_to_pos(8)):
                    return 7
                return 8
        #
        if close[0]<0:
            x = -1
        elif close[0]==0:
            x = 0
        else: x = 1
        #
        if close[1]<0:
            y = -1
        elif close[1]==0:
            y = 0
        else: y = 1
        return self.go_to((x,y))


    def get_action(self, observation):
        if random.uniform(0,1)<0.01:
            return self.space.sample()
        self.set_state(observation)
        if self.healt<self.healt_th and len(self.op_team)>0:
            return self.defensive_move()
        if len(self.op_team)==0:
            return self.search_opponent()
        atk = self.attack_range()
        if (atk!= None):
            return self.attack(atk)
        else:
            return self.chase_closest()





