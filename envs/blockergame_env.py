import numpy as np
import copy 

class BlockerGameEnv():
    def __init__(self, args):
        self.args = args
        self.grid_shape = (4, 7)
        self.blockers = [[0, 2], [4, 6]]
        self._generate_grid()
        
        self.avg_cost_ubound = 0.3
        self.n_agents = 3
        self.observation_spaces = [self.grid_shape[0] * self.grid_shape[1] + 1] * self.n_agents
        self.state_space = self.grid_shape[0] * self.grid_shape[1] * 2
        self.action_spaces = [5] * self.n_agents
        self.n_opponent_actions = 2
    
    def setup(self):
        # generate agents position (maybe randomly)
        #init_pos = np.random.choice(self.grid_shape[1], self.n_agents, replace=False)
        #self.agents_pos_init = [[0, init_pos[i]] for i in range(self.n_agents)]
        
        self.agents_pos_init = [[0, 1], [0, 3], [0, 6]] # static position, applied in CMIX ECML version
        
        self.global_step = 0
        self.returns = None
        self.total_costs = None
        self.peak_violation_sum = None

        return self.n_agents, self.state_space, self.observation_spaces, self.action_spaces, self.n_opponent_actions

    def init_training(self):
        """Called once before training; hook for envs that reset constraint state."""
        pass

    def set_logger(self, logdir):
        self.logdir = logdir
        self.cost_file = open("{}/cost.log".format(logdir), "w", 1)
        self.return_file = open("{}/return.log".format(logdir), "w", 1)
        self.peak_violation_file = open("{}/peak_violation.log".format(logdir), "w", 1)

    def set_scheme(self, scheme):
        self.scheme = scheme
    
    def get_rlinfo(self):
        return self.n_agents, self.state_space, self.observation_spaces, self.action_spaces, self.n_opponent_actions
    

    def _generate_grid(self):
        self.grid = np.zeros(self.grid_shape) # trap or other element
        self.traps = [[1, 3], [1, 6]] # traps
        
        self.cost_matrix = np.array([
            [0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0],
            [1.0, 1.0, 0.1, 1.0, 0.1, 0.1, 1.0],
            [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]])

        print("cost_matrix:", self.cost_matrix)

    def get_grid_obs(self):
        grid_obs = np.ones(self.grid_shape)
        for i in range(self.n_agents):
            grid_obs[self.agents_pos[i][0], self.agents_pos[i][1]] = 0 
        
        return grid_obs

    def reset(self):
        self.win_flag = False
        self.global_step = 0
        
        if self.returns != None:
            avg_cost = self.total_costs / self.returns
            print(self.returns, file=self.return_file)
            print(avg_cost, file=self.cost_file)
            print(self.peak_violation_sum, file=self.peak_violation_file)
        self.returns = 0
        self.total_costs = 0
        self.peak_violation_sum = 0

        self.agents_pos = copy.deepcopy(self.agents_pos_init)
        
        grid_obs = self.get_grid_obs().flatten().tolist()
        cost_obs = self.cost_matrix.flatten().tolist()
        obs_total = grid_obs + cost_obs
        
        obses = []
        for i in range(self.n_agents):
            index = [0] * self.grid_shape[0] * self.grid_shape[1]
            index[self.agents_pos[i][0] * self.grid_shape[1] + self.agents_pos[i][1]] = 1
            obses.append(np.array(index + [self.global_step]))
            
        
        
        state = np.array(obs_total)

        return state, obses
    
    '''
        check whether a cell is vacant(can move in)
    ''' 
    def is_vacant(self, pos):
        x, y = pos
        if x < 0 or x >= self.grid_shape[0]:
            return False
        if y < 0 or y >= self.grid_shape[1]:
            return False
        for blocker in self.blockers:
            if x == self.grid_shape[0] - 1 and y >= blocker[0] and y <= blocker[1]:
                return False
        if pos in self.agents_pos:
            return False

        return True

    def is_safe(self, pos):
        x, y = pos
        if pos in self.traps:
            return False
        
        return True

    def _team_min_manhattan_to_goal(self, agent_positions, blockers):
        """Min over agents of min L1 distance to any bottom-row cell not covered by blockers."""
        row = self.grid_shape[0] - 1
        allowed_ys = []
        for y in range(self.grid_shape[1]):
            blocked = False
            for b in blockers:
                if y >= b[0] and y <= b[1]:
                    blocked = True
                    break
            if not blocked:
                allowed_ys.append(y)
        if not allowed_ys:
            return 0.0
        best = float("inf")
        for p in agent_positions:
            for y in allowed_ys:
                d = abs(p[0] - row) + abs(p[1] - y)
                best = min(best, d)
        return best

    def _potential_phi(self, agent_positions, blockers, scale):
        return -scale * self._team_min_manhattan_to_goal(agent_positions, blockers)
    
    '''
        Blockers moving policy
    '''
    def move_blockers(self):
        # for two blocker 
        if [self.grid_shape[0] - 2, 0] in self.agents_pos:
            self.blockers[0] = [0, 2]
            if [self.grid_shape[0] - 2, self.grid_shape[1] - 1] in self.agents_pos:
                self.blockers[1] = [4, 6]
            else:
                self.blockers[1] = [3, 5]
        else:
            self.blockers = [[1, 3], [4, 6]]
        

    def step(self, actions):
        extra_reward = 0.
        shaping_scale = float(getattr(self.args, "blocker_shaping_scale", 0.0))
        shaping_f = 0.0
        phi_s = 0.0
        
        costs = [0.] * self.n_agents
        peak_violation = 0
        if self.win_flag:
            done_mask = 0
        else:
            done_mask = 1
            if shaping_scale > 0.0:
                pos_before = copy.deepcopy(self.agents_pos)
                blockers_before = copy.deepcopy(self.blockers)
                phi_s = self._potential_phi(pos_before, blockers_before, shaping_scale)
            for i in range(self.n_agents):
                next_pos = copy.copy(self.agents_pos[i])
                if actions[i] == 1:
                    next_pos[0] += 1
                elif actions[i] == 2:
                    next_pos[0] -= 1
                elif actions[i] == 3:
                    next_pos[1] += 1
                elif actions[i] == 4:
                    next_pos[1] -= 1
                elif actions[i] == 0:
                    pass
                
                if actions[i] != 0 and self.is_vacant(next_pos):
                    self.agents_pos[i] = next_pos
                    costs[i] = self.cost_matrix[next_pos[0], next_pos[1]]
                    if actions[i] == 1:
                        extra_reward += 1 / (self.n_agents + 1)
                    elif actions[i] == 2:
                        extra_reward -= 1/ (self.n_agents + 1) 
                if actions[i] != 0 and not self.is_safe(next_pos):
                    peak_violation += 1

                if self.agents_pos[i][0] == self.grid_shape[0] - 1:
                    self.win_flag = True
                    print("Win !!!")
            self.move_blockers()
            if shaping_scale > 0.0:
                phi_sp = self._potential_phi(self.agents_pos, self.blockers, shaping_scale)
                gamma = float(getattr(self.args, "gamma", 0.99))
                shaping_f = gamma * phi_sp - phi_s
        if not self.win_flag:
            #reward = -1
            reward = -1.5 + extra_reward
        else:
            reward = 3
        if shaping_scale > 0.0 and done_mask == 1:
            reward = reward + shaping_f
        
        avg_cost = sum(costs) / self.n_agents
        if done_mask == 1:
            self.total_costs += avg_cost 
            self.returns += 1
            self.peak_violation_sum += peak_violation
        if self.scheme == "simple":
            global_reward = reward
        else:
            global_reward = [reward, min(0, self.avg_cost_ubound - avg_cost - peak_violation)]
        local_rewards = [global_reward] * self.n_agents

        # update state observation
        self.global_step += 1
        
        if self.win_flag:
            self.agents_pos = copy.deepcopy(self.agents_pos_init)  # reset layout after win for next episode

        grid_obs = self.get_grid_obs().flatten().tolist()
        cost_obs = self.cost_matrix.flatten().tolist()
        obs_total = grid_obs + cost_obs
         
       
        obses = []
        for i in range(self.n_agents):
            index = [0] * self.grid_shape[0] * self.grid_shape[1]
            index[self.agents_pos[i][0] * self.grid_shape[1] + self.agents_pos[i][1]] = 1
            obses.append(np.array(index + [self.global_step]))

        state = np.array(obs_total)
        
        return state, obses, local_rewards, global_reward, done_mask
