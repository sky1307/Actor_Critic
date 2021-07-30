from network_topo import Network
import numpy as np


class Environment():
    def __init__(self):
        self.net = Network()
        self.n = self.net.getN()
        self.observation_space = self.getObservation().shape[0]
        self.action_space = self.getAction().shape[0]
        self.observation = self.getObservation()
        self.action = self.getAction()
        self.observation_ = []
        self.reward = 0
        self.done = 0
        self.info = ''

    def reset(self):
        self.__init__()

    def step(self, action):
        self.action = action
        self.set_next_observation(action)
        return self.observation_, self.reward, self.done, self.info

    def set_next_observation(self, action):
        # tinh next state, reward
        self.net.solution = action
        TM_ =  self.net.set_next_trafficMatrix()
        UL_ =  self.net.set_next_utilization()
        self.observation_ = [TM_, UL_]
        self.reward = self.net.getReward()
        return self.observation_

    def getAction(self):
        return np.array(self.net.solution)

    def getReward(self, observation , action):
        reward = 0
        return reward

    def getTrafficMatrix(self):
        traffic = []
        self.net.trafficMatrix = traffic
        return self.net.trafficMatrix

    def getObservation(self):
        state = np.array([self.net.trafficMatrix, self.net.utilization])
        return state



if __name__ == "__main__":
    env = Environment()
    state = env.observation
    for obv in state:
        for i in range(env.n):
            for j in range(env.n):
                print(obv[i][j], end=" ")
            print()
        print()

