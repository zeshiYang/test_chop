'''
time based curriculum environment wrapper
'''
from gym import spaces
import numpy as np
import copy
from IPython import embed

class CLEnv(object):
    '''
    curriculum learning environment for locomotion tasks
    '''
    def __init__(self, env):
        self._env = env
        self._initial_state = self._env.reset()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.current_step = 0
        self._task_t = 1000 #standard settings for gym based tasks

    def render(self, mode="human"):
        return self._env.render(mode)

    def step(self, a):
        state, reward, done, info = self._env.step(a)
        self.current_step +=1
        info['fail'] = done
        if(self.current_step > self._task_t or self._env.num_steps > self._env.episode_length):
            done = True
        if(self._env.name == 'fusion'):
            #print(self._env.name)
            if(self._env.save_state == True):
                phase = self._env.openloop_generator.phase_index(self._env.num_steps * self._env.frame_skip * self._env.time_step)
                #state = self._env.save_sample()
                if(info['rwd_chop1'] < 0.8 or info['rwd_chop2'] < 0.8 or info['pose_rwd']<0.8):
                    sample = self._env.save_sample()
                    self._env.sample_pool.update(sample, phase)
        return state, reward, done, info

    def reset(self):
        state = self._env.reset()
        self._initial_state = state
        self.current_step = 0
        return state

    def get_task_t(self):
        return self._task_t

    def set_task_t(self, t):
        """ Set the max t an episode can have under training mode for curriculum learning
        """
        self._task_t = min(t, 4000)

    def set_new_design(self, design):
        self._env.set_new_design(design)
    def get_current_design(self):
        return self._env.get_current_design()
    def getRenderInfo(self):
        return self._env.getRenderInfo()
    def get_matrix(self):
        return self._env.get_matrix()
    def setKpKd(self, para):
        self._env.setKpKd(para)
    def getKpKd(self):
        return self._env.getKpKd()
    def get_states_buffer(self):
        return self._env.get_states_buffer()

    def set_motionbound(self, bound):
        self._env.set_motionbound(bound)

    def get_motionbound(self):
        return self._env.motionbound

    def set_task_idx(self, para):
        self._env.set_task_idx(para)

    def get_perturb_prob(self):
        return self._env.perturb_prob

    def get_num_task(self):
        return self._env.get_num_task()

    def set_sample_mode(self, mode):
        self._env.sample_mode = mode

    def set_rwd_task(self,para):
        self._env.set_rwd_task(para)

    def render(self):
        self._env.render()

    def close(self):
        pass

    def set_perturb_prob(self, prob):
        self._env.set_perturb_prob(prob)

    def get_episode_length(self):
        return np.min([self._env.get_episode_length(), self._task_t])

    def set_save_state(self, flag = False):
        self._env.save_state = flag

    def get_state_pool(self):
        return self._env.sample_pool

    def load_state(self, path, phase):
        self._env.sample_pool.load(path, phase)

