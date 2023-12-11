import copy
import numpy as np
import random
from itertools import groupby

from tf_agents.environments import utils
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class WaterSortEnv(py_environment.PyEnvironment):

    # Initialize the environment
    def __init__(self, num_bottles:int,  water_level:int, num_empty_bottles:int=2, demo=False):

        super(WaterSortEnv, self).__init__()    # 在继承中，子类继承了父类的所有属性和方法，但父类的构造函数不会自动被调用，所以需要显式调用它来初始化那些继承来的部分

        self.num_bottles = num_bottles                      # total number of bottles
        self.num_empty_bottles = num_empty_bottles          # number of empty bottles
        self.water_level = water_level                      # water level of each bottle
        self.num_colors = num_bottles - num_empty_bottles   # number of colors
        # Add other attributes if needed


        # in the action space, action 0 means the first bottle pour into the second bottle, action 1 means the first bottle pour into the third bottle, and so on.
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=(self.num_bottles*(self.num_bottles-1))-1, name='action')
        
        self._observation_spec = {
            'observation': array_spec.BoundedArraySpec(         # observation is a list of list, each list represents a bottle
                shape=(self.num_bottles, self.water_level), dtype=np.int32, minimum=0, maximum=self.num_colors, name='observation'),
            'action_mask': array_spec.ArraySpec(
                shape=(self.num_bottles*(self.num_bottles-1), ), dtype=bool, name='action_mask')
        }


        # self.bottles_capacity: list[int] capacity of each bottle, should be updated after each action
        # self._state: np.ndarray, shape=(num_bottles, water_level), color of each bottle
        self.winreward = 30
        self._state, self.bottles_capacity, self.num_moves, self.num_actions = self.new_game(True, demo)
        self.initial_state = copy.deepcopy(self._state)             # self._state is an np.ndarray, we need to use deepcopy to copy it
        self.initial_capacity = copy.deepcopy(self.bottles_capacity) 
        self._episode_ended = False
        

        self.demo = demo

    def action_spec(self):
        return self._action_spec
        
    def observation_spec(self):
        return self._observation_spec
    
    def action_mask(self) -> list[int]:
        '''
        对于“倾倒”这一动作：
        1. 对于有颜色的瓶子，只有和顶层相同的颜色才能被倒入，当满了或者加入的颜色和顶层不同的时候，无法执行操作
        2. 对于空瓶子，任何颜色可以被倒入
        3. 当相同颜色在一个瓶子中相邻时, 在执行“倾倒”时也需要注意。假设A为 [蓝,蓝,红,灰], B为[蓝,红,灰](B没有装满), 
            这种情况下从A“倾倒”到B, 只会倒出一个“蓝”到B中。但若B是空瓶子或者有两个“蓝”的空间, 则A中的“蓝”可以被全部倒出。
        '''
        action_mask = [0]*(self.num_bottles*(self.num_bottles-1))   # all False by default
        for i in range(len(action_mask)):
            origin_bottle, dest_bottle = self.action_detail(i)      # get the details of the action, i.e. which bottle to which bottle

            # if the origin bottle is not empty, and the destination bottle is not full
            if self.bottles_capacity[origin_bottle] != 0 and self.bottles_capacity[dest_bottle] != self.water_level:
                if self.bottles_capacity[dest_bottle] == 0:         # if the destination bottle is empty, any color can be poured in
                    action_mask[i] = 1
                else:                                               # if the destination bottle is not empty, only the same color can be poured in
                    origin_bottle_capacity = self.bottles_capacity[origin_bottle]
                    dest_bottle_capacity = self.bottles_capacity[dest_bottle]
                    if self._state[origin_bottle][origin_bottle_capacity-1] == self._state[dest_bottle][dest_bottle_capacity-1]:  # 假设左边是底
                        action_mask[i] = 1

        return action_mask
        
        
    def action_detail(self,action:int) -> tuple[int,int]:
        '''
        通过动作的编号，返回动作的具体细节
        比如5个瓶子, 0表示第一个瓶子倒入第二个瓶子, 1表示第一个瓶子倒入第三个瓶子, 以此类推.
        那么根据动作0, 可以推断出是从第一个瓶子倒入第二个瓶子, 也就是说第一个瓶子是origin_bottle 第二个瓶子是dest_bottle
        '''
        bottle_ind = action // (self.num_bottles-1) # index of bottle
        action_ind = action % (self.num_bottles-1)  # index of action. i.e. 0 means the first action of the bottle

        all_bottles = list(range(self.num_bottles)) # list of all bottles
        all_bottles.pop(bottle_ind)                 # remove itself from the list
        dest_bottle = all_bottles[action_ind]       # index of destination bottle
        origin_bottle = bottle_ind                  # index of origin bottle

        return origin_bottle, dest_bottle
        

    def new_game(self,new:bool,demo:bool=False) -> tuple[np.ndarray, list[int], int, int]:
        '''
        start a new game
        if "new" is True, generate a new game randomly
        if "new" is False, use the initial state
        '''
        
        self._episode_ended = False
        if not new:
            bottles_color = copy.deepcopy(self.initial_state)   # 
            bottles_capacity = copy.deepcopy(self.initial_capacity)
        else:
            if demo:
                bottles_color = self.generate_game_demo()
                bottles_capacity = [self.water_level]*self.num_colors + [0]*self.num_empty_bottles
            else:
                bottles_color, bottles_capacity, maxstep = self.generate_game()
                self.winreward = maxstep + 4
                #print(maxstep)
        
        return bottles_color, bottles_capacity, 0, 0
    

    def generate_game_demo(self) -> np.ndarray:
        '''
        generate a new game 
        '''
        if self.num_bottles == 5:
            game = [[1,1,2,3], [3,2,1,2], [1,3,3,2], [0,0,0,0], [0,0,0,0]]
        elif self.num_bottles == 7:
            game = [[1,2,3,4], [5,5,1,5], [2,2,3,2], [4,1,3,1], [4,3,5,4], [0,0,0,0], [0,0,0,0]]
        bottles_color = np.array(game, dtype=np.int32)

        return bottles_color
    

    def generate_game(self, maxsteps:int=30) -> np.ndarray:
        '''
        generate a new game randomly
        '''
        ########### Initialize the bottles ############
        bottles_color = np.zeros((self.num_bottles, self.water_level), dtype=int)   # no appropriate algorithm yet
        bottles_capacity = [self.water_level]*self.num_colors + [0]*self.num_empty_bottles
        for i in range(self.num_colors):
            bottles_color[i]=[i+1]*self.water_level
            bottles_capacity[i]=self.water_level
        actspace=[]
        for i in range(self.num_bottles):
            for j in range(self.num_bottles):
                if i!=j:
                    actspace.append((i,j))

        end=0
        f=0
        t=0
        for i in range(maxsteps):
            curspace=actspace.copy()
            if i>0:
                curspace.remove((f,t))
                curspace.remove((t,f))
            f,t=random.choice(curspace)
            num=random.choice([1,2])
            while not self.revert_action(bottles_color,bottles_capacity,f,t,num):
                #print("curspace:"+str(len(curspace)))
                curspace.remove((f,t))
                if len(curspace)==0:
                    end=1
                    break
                f,t=random.choice(curspace)
                num=random.choice([1,2])
            if end:
                break
        return bottles_color,bottles_capacity,i
        
    def revert_action(self, bottles_color:np.ndarray, bottles_capacity:list, f:int, t:int, num:int):
        if bottles_capacity[t]==self.water_level:
            return False
        if bottles_capacity[f]==0:
            return False
        process=False
        if bottles_capacity[f]==1:
            process=True
        elif bottles_color[f][bottles_capacity[f]-1]==bottles_color[f][bottles_capacity[f]-2]:
            process=True
        if bottles_capacity[t]>0:
            if bottles_color[t][bottles_capacity[t]-1]==bottles_color[f][bottles_capacity[f]-1] and bottles_capacity[f]!=self.water_level:
                process=False
        if process:
            col=bottles_color[f][bottles_capacity[f]-1]
            cnt=0
            while bottles_capacity[t]<self.water_level and bottles_capacity[f]>0 and cnt<num and bottles_color[f][bottles_capacity[f]-1]==col:
                if bottles_capacity[f]==1:
                    bottles_color[f][bottles_capacity[f]-1]=0
                    bottles_capacity[f]-=1
                    bottles_color[t][bottles_capacity[t]]=col
                    bottles_capacity[t]+=1
                    cnt+=1
                elif bottles_color[f][bottles_capacity[f]-1]==bottles_color[f][bottles_capacity[f]-2]:
                    bottles_color[f][bottles_capacity[f]-1]=0
                    bottles_capacity[f]-=1
                    bottles_color[t][bottles_capacity[t]]=col
                    bottles_capacity[t]+=1
                    cnt+=1
                else:
                    break
            return True
        return False


    def _reset(self):
        '''
        reset the environment
        '''
        #self._state, self.bottles_capacity, self.num_moves, self.num_actions = self.new_game(False, demo=self.demo)   # reset the game but not generate a new game
        self._state, self.bottles_capacity, self.num_moves, self.num_actions = self.new_game(True, demo=self.demo)   # reset the game and generate a new game
        self.initial_state = copy.deepcopy(self._state)                 # reset the initial state
        self._episode_ended = False                                     # reset the episode
        obs = {}
        obs['observation'] = np.array(self._state, dtype=np.int32)
        obs['action_mask'] = np.array(self.action_mask(), dtype=bool)
        #obs['action_mask'] = np.array(self.action_mask(), dtype=np.int32)

        return ts.restart(obs)                                        # return the initial time_step
    

    def get_state(self) -> dict:
        '''
        return the current state
        '''
        obs = {}
        obs['observation'] = np.array(self._state, dtype=np.int32)
        obs['action_mask'] = np.array(self.action_mask(), dtype=bool)
        #obs['action_mask'] = np.array(self.action_mask(), dtype=np.int32)
        return obs
    

    def pour_water(self, origin_bottle:int, dest_bottle:int):
        '''
        pour water from origin bottle to destination bottle
        notice:
        the origin bottle pour water to the destination bottle **as much as possible**
        for example:
            1. A = [0,1,1,2], B = [0,0,0,1]  ----pour A to B---->  A = [0,0,0,2], B = [0,1,1,1]
            2. A = [0,1,1,2], B = [0,1,2,2]  ----pour A to B---->  A = [0,0,1,2], B = [1,1,2,2]
        '''
        self.num_moves += 1
        top_color = self._state[origin_bottle][self.bottles_capacity[origin_bottle]-1]  # get the top color of the origin bottle
        while self.bottles_capacity[origin_bottle] > 0 and self.bottles_capacity[dest_bottle] < self.water_level:
            if self._state[origin_bottle][self.bottles_capacity[origin_bottle]-1] == top_color:    # if the color is the same as the top color
                self._state[origin_bottle][self.bottles_capacity[origin_bottle]-1] = 0             # continue to pour
                self.bottles_capacity[origin_bottle] -= 1
                self.bottles_capacity[dest_bottle] += 1
                self._state[dest_bottle][self.bottles_capacity[dest_bottle]-1] = top_color
            else:
                break


    def is_game_over(self) -> str:
        '''
        check if the game is over.
        conditions:
            1. if all action_mask are False, then no actions can be taken and the game is over
            2. otherwise, if all bottles are full except two empty bottles
            and if every single full bottle contains one color, then the game is over
        '''
        action_mask = self.action_mask()
        if not any(action_mask):        # if all action_mask are False, then no actions can be taken and the game is over
            return 'lose'
        else:
            if self.num_actions >= self.winreward:  # if the number of actions exceeds the maximum number of actions, then the game is over
                return 'lose'
            
            elif self.bottles_capacity.count(0) == self.num_empty_bottles:    # if all bottles are full except two empty bottles
                full_bottles = [i for i in range(self.num_bottles) if self.bottles_capacity[i] == self.water_level] # list of full bottles
                for bottle in full_bottles:                         # if every single full bottle contains one color, then the game is over      
                    if not self.all_equal(self._state[bottle]):
                        return 'not over'
                    
                return 'win'
            
            else:
                return 'not over'


    def all_equal(self, iterable:list|np.ndarray) -> bool:
        g = groupby(iterable)
        return next(g, True) and not next(g, False)


    def _step(self, action:int):
        '''
        take one step
        '''
        # TODO: add reward
        self.num_actions += 1
        if self._episode_ended:
            return self.reset()
        
        action_mask = self.action_mask()    # get the action mask

        #invalid_action = 1
        invalid_action = 0
        if action_mask[action]:             # if the action is valid
            #invalid_action = 0
            origin_bottle, dest_bottle = self.action_detail(action)
            self.pour_water(origin_bottle, dest_bottle)

        game_result = self.is_game_over()   # if the game is over
        if game_result == 'win':
            # TODO: add reward

            obs = {}
            obs['observation'] = np.array(self._state, dtype=np.int32)
            obs['action_mask'] = np.array(self.action_mask(), dtype=bool)
            #obs['action_mask'] = np.array(self.action_mask(), dtype=np.int32)
            self._episode_ended = True
            return ts.termination(obs, reward=self.winreward)
        
        elif game_result == 'lose': # add other conditions if needed
            # TODO: add reward

            obs = {}
            obs['observation'] = np.array(self._state, dtype=np.int32)
            obs['action_mask'] = np.array(self.action_mask(), dtype=bool)
            #obs['action_mask'] = np.array(self.action_mask(), dtype=np.int32)
            self._episode_ended = True
            return ts.termination(obs, reward=-(self.winreward))
        
        elif game_result == 'not over':
            # TODO: add reward
            obs = {}
            obs['observation'] = np.array(self._state, dtype=np.int32)
            obs['action_mask'] = np.array(self.action_mask(), dtype=bool)
            #obs['action_mask'] = np.array(self.action_mask(), dtype=np.int32)
            return ts.transition(obs, reward=-1-invalid_action, discount=1.0)   # change if needed
        
        else:
            raise ValueError('game_result is not valid')

