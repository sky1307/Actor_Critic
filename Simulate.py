from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker import Worker

from Simulator.Sensor_Node.node import Node
from Simulator.Network.network import Network
from Simulator.Mobile_Charger.mobile_charger import MobileCharger
from Simulator.parameter import SIM_duration, NODE_e_thresh_ratio

import random
import pandas as pd
from ast import literal_eval

import csv
from scipy.stats import sem, t, tmean

import os

os.system("rm log/weight_record/*")
os.system("rm log/Worker*")
with open("log/weight_record/loss.csv", "w") as dumpfile:
    dumpfile.write("Time\tWorker\tTD\tmu\tPLoss\tELoss\tVLoss\n")

experiment_type = input('experiment_type: ')  # ['node', 'target', 'MC', 'prob', 'package', 'cluster']
df = pd.read_csv("data/" + experiment_type + ".csv")
experiment_index = int(input('experiment_index: '))  # [0..6]

# | Experiment_index  | 0   | 1   | 2     | 3   | 4   | 5     | 6     | 7   | 8    |
# |------------------ |-----|-----|-------|-----|-----|-------|-------|-----|------|
# | node              |`300`|`350`|__400__|`450`|`500`| 550   | 600   | 650 | 700  |
# | target            |`200`|`250`|__300__|`350`|`400`| 450   | 500   | 550 | 600  |
# | MC                | 1   |`2`  |__3__  | `4` |`5`  |`6`    | 7     | 8   | 9    |
# | prob              | 0.1 | 0.2 | 0.3   |`0.4`|`0.5`|__0.6__|`0.7`  |`0.8`| 0.9  |
# | package           | 400 | 450 | 500   | 550 |`600`|`650`  |__700__|`750`|`800` |
# | cluster           | 40  | 45  | 50    | 55  |`60` |`65`   | `70`  |`75` |__80__|

output_file = open("log/Actor_Critic_Kmeans.csv", "w")
result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
result.writeheader()

com_ran = df.commRange[experiment_index]
prob = df.freq[experiment_index]
nb_mc = df.nb_mc[experiment_index]
theta = df.q_alpha[experiment_index]
clusters = df.charge_pos[experiment_index]
package_size = df.package[experiment_index]

life_time = []
for nb_run in range(1):
    random.seed(nb_run)

    energy = df.energy[experiment_index]
    energy_max = df.energy[experiment_index]
    node_pos = list(literal_eval(df.node_pos[experiment_index]))

    list_node = []
    for i in range(len(node_pos)):
        location = node_pos[i]
        node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                    energy_thresh=NODE_e_thresh_ratio * energy, prob=prob)
        list_node.append(node)

    # Global optimizer
    nb_state_feature = nb_mc * 3 + (clusters + 1) * 4
    global_Optimizer = Server(nb_action=clusters + 1, nb_state_feature=nb_state_feature, name="Global Optimizer")
    mc_list = []
    optimizer_list = []
    for id in range(nb_mc):
        # optimizer = Actor_Critic(id=id, nb_action=clusters, alpha=alpha)
        optimizer = Worker(Server_object=global_Optimizer, name="worker_" + str(id), id=id, theta=theta)
        optimizer_list.append(optimizer)  # List worker
        mc = MobileCharger(id, energy=df.E_mc[experiment_index], capacity=df.E_max[experiment_index],
                           e_move=df.e_move[experiment_index],
                           e_self_charge=df.e_mc[experiment_index], velocity=df.velocity[experiment_index],
                           optimizer=optimizer)
        mc_list.append(mc)
    target = [int(item) for item in df.target[experiment_index].split(',')]
    net = Network(list_node=list_node, mc_list=mc_list, target=target,
                  server=global_Optimizer, package_size=package_size, nb_charging_pos=clusters)
    print(
        "experiment {} #{}:\n\tnode: {}, target: {}, prob: {}, mc: {}, alpha: {}, cluster: {}, package_size: {}".format(
            experiment_type, experiment_index, len(net.node), len(net.target), prob, nb_mc, theta, clusters,
            package_size))
    file_name = "log/Actor_Critic_Kmeans_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
    try:
        temp = net.simulate(max_time=SIM_duration, file_name=file_name)
        life_time.append(temp[0])
        result.writerow({"nb_run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

    finally:
        # free memory space
        print("Free memory space")
        del global_Optimizer
        for optimizer in optimizer_list:
            del optimizer

confidence = 0.95
h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
result.writerow({"nb_run": tmean(life_time), "lifetime": h, "dead_node": 0})
