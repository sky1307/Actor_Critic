import random

import torch
import Simulator.parameter as para

def synchronize(server, mc_list):
    """
    This function synchronize all MC's networks with Server's
    :param server: cloud
    :param mc_list: list of MC
    """
    with open("log/weight_record/actor_param.txt", "a+") as dumpfile:
        dumpfile.write("One update step\n")

    for MC in mc_list:
        MC.optimizer.actor_net.load_state_dict(server.actor_net.state_dict())
        MC.optimizer.critic_net.load_state_dict(server.critic_net.state_dict())


def update_gradient(server, MC_networks, debug=True):
    """
    :param server: cloud
    :param MC_networks: is ONE MC's networks, or Worker.net
    """
    for MC_partial_net, SV_partial_net, learning_rate in zip(MC_networks, server.net, server.lr):
        for serverParam, MCParam in zip(SV_partial_net.parameters(), MC_partial_net.parameters()):
            if not torch.isnan(MCParam.grad).any():
                if debug:
                    debug_weights_update(serverParam.data, MCParam.grad)
                serverParam.data -= learning_rate * MCParam.grad

    server.actor_lr = server.actor_lr * server.decay_lr if server.actor_lr > para.A3C_serverActor_lr else para.A3C_serverActor_lr
    server.critic_lr = server.critic_lr * server.decay_lr if server.critic_lr > para.A3C_serverCritic_lr else para.A3C_serverCritic_lr
    server.body_lr = server.body_lr * server.decay_lr if server.body_lr > para.A3C_serverBody_lr else para.A3C_serverBody_lr


def zero_net_weights(net):
    for net_param in net.parameters():
        net_param.data = torch.rand_like(net_param.data) * random.gauss(0, 0.01)


def debug_weights_update(param_data, grad):
    with open("log/weight_record/param.txt", "a+") as dumpfile:
        dumpfile.write(str(torch.sum(param_data)) + "\t" + str(torch.sum(grad)) + "\n")


if __name__ == "__main__":
    a = [1, 2, 3, 4]
    a_torch = torch.Tensor(a)
    print(torch.rand_like(a_torch) * random.gauss(0, 0.01))
