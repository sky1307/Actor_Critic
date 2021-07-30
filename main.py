import numpy as np
from acvp import Agent
from environment.environment import Environment

if __name__ == '__main__':
    env = Environment()
    agent = Agent(input_dims = env.observation_space, env = env, 
            n_actions = env.action_space.shape[0])

    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps +=1
        agent.learn()
        agent.load_models()
        evaluate = True

    while True:
        env.reset()
        observation = env.observation
        while True:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

            agent.save_models()

