import torch
import numpy as np
from Optimizer.A3C.Server_method import update_gradient
from Optimizer.A3C.heuristic import H_charging_time_func, H_get_heuristic_policy


def get_nearest_charging_pos(current_location, charging_pos_list):
    mc_pos = torch.Tensor(current_location)
    A = torch.Tensor(charging_pos_list)
    distance_list = torch.sqrt(torch.sum(torch.pow(mc_pos - A, 2), dim=1))
    min_index = torch.argmin(distance_list)
    return charging_pos_list[min_index]


# TODO: re-define the reward
def reward_function(Worker, mc, network, time_stamp):
    e_list = []
    for each_node in network.node:
        if each_node.energy > 0:
            e_list.append(each_node.energy)

    print("Average energy of living nodes: " + str(np.mean(np.array(e_list))))
    if Worker.step < 100:
        return -1 / min(e_list)
    else:
        return min(e_list)


def TERMINAL_STATE(state_tensor):
    # return state_tensor[0] == para.depot[0] and state_tensor[1] == para.depot[1]
    return False


def extract_state_tensor(worker, net):
    def normalization(input, max, min):
        if np.isscalar(input):
            return [-1 + (input - min) * (1 - -1) / (max - min)]
        else:
            input = np.array(input)
            normalized_np = -1 + (input - min) * (1 - -1) / (max - min)
            return list(normalized_np)

    pos_max = 999
    pos_min = 0

    # Implement here
    MC = net.mc_list[worker.id]
    MC_location = [MC.current[0], MC.current[1]]  # get x, y coordination
    MC_location = normalization(MC_location, pos_max, pos_min)  # location normalization
    MC_energy = normalization(MC.energy, MC.capacity, 0)  # energy normalization
    MC_info = MC_location + MC_energy  # concat list

    MC_info_tensor = torch.Tensor(MC_info)
    MC_info_tensor = torch.flatten(MC_info_tensor)  # flatten (3) [x, y, E]
    MC_info_tensor.requires_grad = False

    charge_pos_info = []
    charge_pos = net.charging_pos
    for mc in net.mc_list:
        if mc.id != MC.id:
            x_mc, y_mc = get_nearest_charging_pos(mc.current, charge_pos)
            MC_location = normalization([x_mc, y_mc], pos_max, pos_min)  # location normalization
            e_mc = normalization(mc.energy, mc.capacity, 0)  # energy normalization
            charge_pos_info.append(MC_location + e_mc)  # concat list

    charge_pos_tensor = torch.Tensor(charge_pos_info)
    charge_pos_tensor = torch.flatten(charge_pos_tensor)  # flatten (3 x nb_mc - 3)
    charge_pos_tensor.requires_grad = False

    partition_info = []
    for i, pos in enumerate(charge_pos):
        charge_pos = normalization(pos, pos_max, pos_min)
        min_E = float('inf')
        max_e = float('-inf')

        for index_node in net.index_node_in_cluster[i]:
            node = net.node[index_node]
            if node.energy > 0:
                if node.energy < min_E:
                    min_E = node.energy
                if node.avg_energy > max_e:
                    max_e = node.avg_energy
        if min_E == float('inf'):
            min_E = 0
            max_e = 0
        min_E = normalization(min_E, 10, 0)
        max_e = normalization(max_e, 10, 0)
        partition_info.append(charge_pos + min_E + max_e)  # concat list

    partition_info_tensor = torch.Tensor(partition_info)
    partition_info_tensor = torch.flatten(partition_info_tensor)  # 4 x nb_action
    partition_info_tensor.requires_grad = False

    state = torch.cat([MC_info_tensor, charge_pos_tensor, partition_info_tensor])
    # return Tensor form of the state
    return state  # 3 x nb_mc, 4 x nb_charging_pos


# TODO: get state from network (new - Nguyen Thanh Long)
def extract_state_tensor_v2(worker, net):
    return None


def charging_time_func(mc=None, net=None, action_id=None, time_stamp=0, theta=0.1):
    return H_charging_time_func(mc=mc, net=net, action_id=action_id, time_stamp=time_stamp, theta=theta)
    # return 150


# TODO: implement heuristic policy (Nguyen Thanh Long)
def get_heuristic_policy(net=None, mc=None, worker=None, time_stamp=0):
    return H_get_heuristic_policy(net=net, mc=mc, worker=worker, time_stamp=time_stamp)


def one_hot(index, size):
    one_hot_vector = np.zeros(size)
    one_hot_vector[index] = 1
    one_hot_vector = torch.Tensor(one_hot_vector)
    one_hot_vector.requires_grad = False
    return one_hot_vector


def tensor2value(tensor):
    return tensor.detach().numpy()


def asynchronize(Worker, Server, time_step=None):  # MC sends gradient to Server
    """
    :param time_step:
    :param Worker: current MC's optimizer (self)
    :param Server: cloud
    This function perform asynchronize update to the cloud
    """
    print(f"Worker id_{Worker.id} asynchronized with len(buffer): {len(Worker.buffer)}")
    if len(Worker.buffer) >= 2:
        Worker.accumulate_gradient(time_step=time_step)
        # networks = (Worker.actor_net, Worker.critic_net)
        networks = Worker.net
        update_gradient(Server, networks)

        # clean gradient
        Worker.reset_grad()
        # clear record
        lastBuffer = Worker.buffer[-1]
        Worker.buffer.clear()
        # restore current state for next use
        Worker.buffer.append(lastBuffer)
        return True
    else:
        print(f"Worker id_{Worker.id} has nothing to asynchronize")
        return False


def all_asynchronize(MCs, Server, moment=None):
    """
    :param moment: current time step
    :param MCs: list of MC
    :param Server: cloud
    """
    print("All asynchronize!")
    any_news = []
    for MC in MCs:
        any_news.append(asynchronize(Worker=MC.optimizer, Server=Server, time_step=moment))

    return True in any_news


if __name__ == "__main__":
    T = [True, False, True]
    print(True in T)
