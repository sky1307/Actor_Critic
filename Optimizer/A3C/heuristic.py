import random

from Simulator.Sensor_Node.node_method import find_receiver
from scipy.spatial import distance
import torch
import numpy as np
import Simulator.parameter as para


def H_charging_time_func(mc=None, net=None, action_id=None, time_stamp=0, theta=0.1):
    """
    :param mc: mobile charger
    :param net: network
    :param action_id: index of charging position
    :param time_stamp: current time stamp
    :param theta: hyper-parameter
    :return: duration time which the MC will stand charging for nodes
    """
    charging_position = net.charging_pos[action_id]
    time_move = distance.euclidean(mc.current, charging_position) / mc.velocity
    energy_min = net.node[0].energy_thresh + theta * net.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    # print(charging_position, len(net.request_id))
    for requesting_node in net.request_id:
        node = net.node[requesting_node]
        d = distance.euclidean(charging_position, node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in net.mc_list:
            if other_mc.id != mc.id and other_mc.get_status() == "charging":
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2) * (other_mc.end_time - time_stamp)
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node.id, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node.id, p, p1))
    t = []

    for index, p, p1 in s1:
        t.append((energy_min - net.node[index].energy + time_move * net.node[index].avg_energy - p1) / (
                p - net.node[index].avg_energy))
    for index, p, p1 in s2:
        t.append((energy_min - net.node[index].energy + time_move * net.node[index].avg_energy - p1) / (
                p - net.node[index].avg_energy))
    dead_list = []
    for item in t:
        nb_dead = 0
        for index, p, p1 in s1:
            temp = net.node[index].energy - time_move * net.node[index].avg_energy + p1 + (
                    p - net.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        for index, p, p1 in s2:
            temp = net.node[index].energy - time_move * net.node[index].avg_energy + p1 + (
                    p - net.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    else:
        return 0


def H_get_heuristic_policy(net=None, mc=None, worker=None, time_stamp=0):
    energy_factor = torch.ones_like(torch.Tensor(worker.action_space))
    priority_factor = torch.ones_like(torch.Tensor(worker.action_space))
    target_monitoring_factor = torch.ones_like(torch.Tensor(worker.action_space))
    self_charging_factor = torch.ones_like(torch.Tensor(worker.action_space))
    for action_id in worker.action_space:
        temp = heuristic_function(net=net, mc=mc, optimizer=worker, action_id=action_id, time_stamp=time_stamp)
        energy_factor[action_id] = temp[0]
        priority_factor[action_id] = temp[1]
        target_monitoring_factor[action_id] = temp[2]
        self_charging_factor[action_id] = temp[3]
    energy_factor = energy_factor / (torch.sum(energy_factor) + 1e-6)

    priority_factor = priority_factor / (torch.sum(priority_factor) + 1e-6)

    target_monitoring_factor = target_monitoring_factor / (torch.sum(target_monitoring_factor) + 1e-6)

    self_charging_factor = self_charging_factor / (torch.sum(self_charging_factor) + 1e-6)

    H_policy = (energy_factor + priority_factor + target_monitoring_factor - self_charging_factor)

    H_policy = torch.Tensor(H_policy)
    H_policy = para.A3C_deterministic_factor * (H_policy - torch.mean(H_policy))
    G = torch.exp(H_policy)
    H_policy = G / torch.sum(G)
    H_policy.requires_grad = False

    if torch.isnan(H_policy).any():
        print(energy_factor)
        print(priority_factor)
        print(target_monitoring_factor)
        print(self_charging_factor)
        print("Heuristic policy contains Nan value")
        exit(120)

    return H_policy  # torch tensor size = #nb_action


def heuristic_function(net=None, mc=None, optimizer=None, action_id=0, time_stamp=0, receive_func=find_receiver):
    if action_id == optimizer.nb_action - 1:
        return 0, 0, 0, 0
    theta = optimizer.charging_time_theta
    charging_time = H_charging_time_func(mc, net, action_id=action_id, time_stamp=time_stamp,
                                         theta=theta)
    w, nb_target_alive = get_weight(net=net, mc=mc, action_id=action_id, charging_time=charging_time,
                                    receive_func=receive_func)
    p = get_charge_per_sec(net=net, action_id=action_id)
    p_hat = p / np.sum(p)
    p_total = get_total_charge_per_sec(net=net, action_id=action_id)
    E = np.asarray([net.node[request["id"]].energy for request in net.request_list])
    e = np.asarray([request["avg_energy"] for request in net.request_list])
    third = nb_target_alive / len(net.target)
    second = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    forth = (mc.capacity - (mc.energy - charging_time * p_total)) / mc.capacity
    # print('At {}s, HF for action {}: {}, {}, {}, {}'.format(time_stamp, action_id, first, second, third, forth))
    if mc.energy - charging_time * p_total < 0:
        return 0, 0, 0, 1
    return first, second, third, forth


def get_weight(net, mc, action_id, charging_time, receive_func=find_receiver):
    p = get_charge_per_sec(net, action_id)
    all_path = get_all_path(net, receive_func)
    time_move = distance.euclidean(mc.current, net.charging_pos[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in net.request_list]
    for request_id, request in enumerate(net.request_list):
        temp = (net.node[request["id"]].energy - time_move * request["avg_energy"]) + (
                p[request_id] - request["avg_energy"]) * charging_time
        if temp < 0:
            list_dead.append(request["id"])
    for request_id, request in enumerate(net.request_list):
        nb_path = 0
        for path in all_path:
            if request["id"] in path:
                nb_path += 1
        w[request_id] = nb_path
    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    for path in all_path:
        if para.base in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive


def get_charge_per_sec(net=None, action_id=None):
    return np.asarray(
        [para.alpha / (distance.euclidean(net.node[request["id"]].location,
                                          net.charging_pos[action_id]) + para.beta) ** 2 for
         request in net.request_list])


def get_total_charge_per_sec(net=None, action_id=None):
    p = np.asarray([para.alpha / (distance.euclidean(node.location,
                                                     net.charging_pos[action_id]) + para.beta) ** 2 for
                    node in net.node])
    return np.sum(p)


def get_path(net, sensor_id, receive_func=find_receiver):
    path = [sensor_id]
    if distance.euclidean(net.node[sensor_id].location, para.base) <= net.node[sensor_id].com_ran:
        path.append(para.base)
    else:
        receive_id = receive_func(net=net, node=net.node[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id, receive_func))
    return path


def get_all_path(net, receive_func=find_receiver):
    list_path = []
    for sensor_id, target_id in enumerate(net.target):
        list_path.append(get_path(net, sensor_id, receive_func))
    return list_path


if __name__ == "__main__":
    a = torch.rand(15) * random.gauss(0.011, 0.005)
    d = 2 * (a - torch.mean(a)) / torch.std(a)
    c = torch.exp(d)
    b = c / torch.sum(c)
    print(a)
    print("new way", b)
    # c = torch.exp(a)
    # b = c/torch.sum(c)
    # print("old way", b)
