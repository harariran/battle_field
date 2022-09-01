import random

from agents import DecisionMaker

class Simple_DM(DecisionMaker):
    def __init__(self, action_space, health_th=0.5 , red_team=False):
        self.space = action_space
        self.healt = None
        self.healt_th = health_th
        self.walls = []
        self.my_team = []
        self.op_team = []
        self.is_red_team = red_team
        self.counter = 0
        self.enemy_orientation = None # remembers the direction of an enemy


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

    # def defensive_move(self):
    #     if self.is_red_team:
    #         return 4
    #     else: return 8
    #     #todo set defensive better

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
        if self.is_red_team:
            if ((1, 0) in self.walls) or ((2, 0) in self.walls):
                return 0
            else:
                return random.choice([3, 8, 7, 11])
        else:
            if ((-1,0) in self.walls) or ((-2,0) in  self.walls):
                return 12
            else:
                return random.choice([1,4,5,9])

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
        close_pos = (0,0)
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

    def group(self):
        if len(self.my_team)==0:
            return self.defensive_move()
        x = 0
        y = 0
        for i, _ in self.my_team:
            if i[0]!=0:
                x+= i[0]/abs(i[0])
            if i[1]!=0:
                y+= i[1]/abs(i[1])
        if x!=0:
            x = int(x/abs(x))
        if y!=0:
            y = int(y/abs(y))
        go_direction = (x,y)
        if x==0:
            if y==0:
                return 6
            else:
                dbl_go= (x,y*2)
                if self.check_my_team(dbl_go):
                  return self.go_to(go_direction)
        else:
            if y==0:
                dbl_go = (x * 2, y)
                if self.check_my_team(dbl_go):
                    return self.go_to(go_direction)
            else:
                if self.check_my_team(go_direction):
                    if self.check_my_team((0,y*2)):
                        return self.go_to((x*2,0))
                    else:
                        return self.go_to((0,y*2))
                else:
                    return self.go_to(go_direction)
        return 6 # not in use


    def get_action(self, observation):
        self.counter+=1
        if self.counter==50:
            self.counter=0
            self.is_red_team = not self.is_red_team
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

    def get_Low_level_action(self, observation, HL_action):
        self.set_state(observation)
        self.counter+=1
        if self.counter==50:
            self.counter=0
            self.is_red_team = not self.is_red_team
        if HL_action==0: # group
            return self.group()
        elif HL_action==1: # attack (search, chase, attack)
            if len(self.op_team) == 0:
                if (self.enemy_orientation==None) or self.last_orientation>self.counter-20:
                    return self.search_opponent()
                else:
                    return self.enemy_orientation
            atk = self.attack_range()
            if atk != None:
                return self.attack(atk)
            else:
                self.enemy_orientation = self.chase_closest()
                self.last_orientation = self.counter
                return self.enemy_orientation
        elif HL_action == 2:  # defensive move
            return self.defensive_move()
        else: # if action not 'legal' do nothing
            return 6





