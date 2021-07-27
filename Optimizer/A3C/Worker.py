import torch
import numpy as np
import Simulator.parameter as para
from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker_method import reward_function, TERMINAL_STATE, \
    extract_state_tensor, charging_time_func, get_heuristic_policy, \
    extract_state_tensor_v2, one_hot, tensor2value

import csv


class Worker(Server):  # Optimizer
    def __init__(self, Server_object, name, id, theta):
        super(Worker, self).__init__(nb_state_feature=Server_object.nb_state_feature,
                                     nb_action=Server_object.nb_action,
                                     name=name)

        self.id = id
        self.step = 0

        self.beta_entropy = para.A3C_beta_entropy
        self.gamma = para.A3C_gamma
        self.alpha_H = para.A3C_alpha_heuristic
        self.theta_H = para.A3C_decay_heuristic
        self.charging_time_theta = theta  # charging time function hyper-parameter

        self.buffer = []
        self.action_space = [i for i in range(self.nb_action)]

    def create_experience(self, state, action,
                          policy_prob, behavior_prob,
                          reward=None):
        experience = {
            "step": self.step,              # record step t
            "reward": reward,               # record reward observe in this state
            "state": state,                 # record state vector t
            "action": action,               # record action at step t
            "policy_prob": policy_prob,     # record policy prob(action t)
            "behavior_prob": behavior_prob  # record behavior prob(action t)
        }
        self.step += 1
        return experience

    def policy_loss_fn(self, policy, action, temporal_diff):
        return - torch.sum(temporal_diff[0] * torch.log(policy) * torch.Tensor(one_hot(size=self.nb_action,
                                                                                       index=action)))

    def entropy_loss_fn(self, policy):
        return self.beta_entropy * torch.sum(policy * torch.log(policy))

    def value_loss_fn(self, value, reward):
        return (1/2) * torch.pow(reward - value, 2)

    def accumulate_gradient(self, time_step=None, debug=True):
        # i.e. (R,S,A) = [(R0,S0,A0),(R1,S1,A1),(R2,S2,A2)]
        R = 0 if TERMINAL_STATE(self.buffer[-1]["state"]) \
            else self.get_value(self.buffer[-1]["state"])      # R[2]

        t = len(self.buffer) - 1        # t = 2
        M = [self.buffer[i]["policy_prob"] / self.buffer[i]["behavior_prob"] for i in range(t)]
        mu = 1
        for i in range(t):              # mu = pi(A0|S0)/b(A0|S0) * pi(A1|S1)/b(A1|S1)
            mu *= M[i]

        for i in range(t):              # 0, 1
            j = t - 1 - i               # 1, 0
            R = self.buffer[j]["reward"] + self.gamma * R

            state_vector = self.buffer[j]["state"]
            value = self.get_value(state_vector)
            policy = self.get_policy(state_vector)

            value_loss = self.value_loss_fn(value=value, reward=R)
            value_loss.backward(retain_graph=True)

            tmp_diff = R - value
            truncated_mu = min(mu, para.A3C_clipping_mu)
            policy_loss = self.policy_loss_fn(policy=policy,
                                              temporal_diff=tensor2value(tmp_diff),
                                              action=self.buffer[j]["action"]) * truncated_mu

            entropy_loss = self.entropy_loss_fn(policy=policy)
            policy_total_loss = policy_loss + entropy_loss
            policy_total_loss.backward(retain_graph=True)

            if debug:
                with open("log/weight_record/loss.csv", "a+") as dumpfile:
                    dumpfile.write(f"{time_step}\t{self.id}\t{tensor2value(tmp_diff)[0]}\t{truncated_mu}\t{tensor2value(policy_loss)}\t"
                                   f"{tensor2value(entropy_loss)}\t{tensor2value(value_loss)[0]}\n")
            mu /= M[j]

    def reset_grad(self):
        for partial_net in self.net:
            partial_net.zero_grad()

    def get_action(self, network=None, mc=None, time_stamp=None):
        R = 0
        if self.step != 0:
            R = reward_function(Worker=self, mc=mc, network=network, time_stamp=time_stamp)

        state_tensor = extract_state_tensor(self, network)
        policy = self.get_policy(state_tensor)
        if torch.isnan(policy).any():
            FILE = open("debug.txt", "w")
            FILE.write(np.array2string(tensor2value(state_tensor)))
            FILE.write("\n")
            FILE.write(np.array2string(tensor2value(policy)))
            FILE.close()
            print("Error Nan policy")
            exit(100)

        heuristic_policy = get_heuristic_policy(net=network, mc=mc, worker=self, time_stamp=time_stamp)

        behavior_policy = (1 - self.alpha_H) * policy + self.alpha_H * heuristic_policy
        action = np.random.choice(self.action_space, p=tensor2value(behavior_policy))

        # apply decay on alpha_H
        self.alpha_H *= self.theta_H

        # get charging time
        if action == self.nb_action - 1:
            charging_time = (mc.capacity - mc.energy) / mc.e_self_charge
        else:
            charging_time = charging_time_func(mc=mc, net=network, action_id=action, time_stamp=time_stamp,
                                          theta=self.charging_time_theta)

        with open(f"log/Worker_{self.id}.csv", mode="a+") as dumpfile:
            dumpfile_writer = csv.writer(dumpfile)
            if self.step != 0:
                dumpfile_writer.writerow([
                    self.step, time_stamp, R, tensor2value(state_tensor), action, tensor2value(policy[action]),
                    tensor2value(heuristic_policy[action]), tensor2value(behavior_policy[action]),
                    charging_time
                ])
            else:
                dumpfile_writer.writerow(["step", "time_stamp", "reward", "state", "action",
                                          "policy_prob", "heuristic_prob", "behavior_prob", "charging_time"])

        # record all transitioning and reward
        self.buffer.append(
            self.create_experience(
                state=state_tensor, action=action,
                policy_prob=tensor2value(policy[action]), behavior_prob=tensor2value(behavior_policy[action]),
                reward=R
            )
        )

        return action, charging_time


if __name__ == "__main__":
    action_space = torch.Tensor([1, 2, 3, 4])
    # print(tensor2value(action_space[1]))
    # a = torch.Tensor(action_space)
    # print(a[1])

    b = torch.Tensor([3])
    b.requires_grad = True
    a = torch.Tensor([4])
    a.requires_grad = True
    with torch.no_grad():
        c = 3 * b + a
    c.backward(retain_graph=True)
    # c.backward(retain_graph=True)
    print(b.grad)
    print(a.grad)
    # print(b.detach().numpy()[0])
    # alpha_H = 0.8
    # a = torch.Tensor([0.1, 0.2, 0.3, 0.4])
    # b = np.array([4,3,1,2])
    # c = np.exp(b)/np.sum(np.exp(b))
    # d = (1 - alpha_H) * a + alpha_H * c
    # print(d)
    # print(np.random.choice(action_space, p=d.detach().numpy()))

    # b = [1,2,3,4,5]
    # c = torch.ones_like(torch.Tensor(b))
    # print(c)




