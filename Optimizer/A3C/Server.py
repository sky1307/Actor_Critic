import torch
import torch.nn as nn
import Simulator.parameter as para
from Optimizer.A3C.Server_method import update_gradient, zero_net_weights


class Server(nn.Module):
    def __init__(self, nb_state_feature, nb_action, name):
        super(Server, self).__init__()

        self.body_net = nn.Sequential(
            nn.Linear(in_features=nb_state_feature, out_features=256),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=512),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.Sigmoid(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=128),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=nb_action),
            nn.Softmax()
        )
        zero_net_weights(self.actor_net)

        self.critic_net = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.Sigmoid(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=128),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        zero_net_weights(self.critic_net)

        self.actor_lr = para.A3C_start_Actor_lr
        self.critic_lr = para.A3C_start_Critic_lr
        self.body_lr = para.A3C_start_Body_lr

        self.decay_lr = para.A3C_decay_lr

        self.net = [self.body_net, self.actor_net, self.critic_net]
        self.lr = [self.body_lr, self.actor_lr, self.critic_lr]

        self.nb_state_feature = nb_state_feature
        self.nb_action = nb_action

        self.name = name

    def get_policy(self, state_vector):
        body_out = self.body_net(state_vector)
        return self.actor_net(body_out)

    def get_value(self, state_vector):
        body_out = self.body_net(state_vector)
        return self.critic_net(body_out)

    def update_gradient(self, MC_networks):
        update_gradient(self, MC_networks)


if __name__ == "__main__":
    # a = (1, 2, 3, 4, 5)
    # b = [4, 3, 5, 2, 6]
    # c = zip(a, b)
    # for x, y in c:
    #     print(x, y)
    a = torch.Tensor([1,2,3,4,5])
    b = torch.Tensor([5,8,1,2,3])
    e = torch.Tensor([7,3,5,8,1])
    c = torch.reshape(torch.cat([a,b,e]),[1, 1, 3, 5])
    # c = torch.dot(a, b)
    print(c)
    conv1 = torch.nn.Conv1d(in_channels=1, out_channels=5, kernel_size=(3,2))
    d = conv1(c)
    print(d)
    # c = torch.cat([a, b], 0)
    # print(c)
