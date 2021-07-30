import os
import csv


class Network():
    def __init__(self):
        self.x = []
        self.y = []
        self.label_node = []
        self.weights = []
        self.bw = []
        self.delay = []
        self.label_edge = []
        self.trafficMatrix = []
        self.n = 12
        self.utilization = []
        self.solution =[]
        self.loadTopo()

    def getTopo(self):
        return [self.weights, self.bw, self.delay]

    def getN(self):
        return self.n

    def loadTopo(self):
        # load Node

        self.file_topo_node = f"{os.getcwd()}/topo/abilene_tm_node.csv"
        # self.file_topo_node = os.path.join('AC_VP/ddpg/environment/topo/abilene_tm_node.csv')
        with open(self.file_topo_node) as csv_file:
            csv_node = csv.reader(csv_file, delimiter=' ')
            line_count = -1
            for row in csv_node:
                if line_count == -1:
                    line_count += 1
                else:
                    self.label_node.append(row[0])
                    self.x.append(float(row[1]))
                    self.y.append(float(row[2]))
                    line_count += 1
        self.n = line_count

        # load edge
        self.file_topo_edge = f"{os.getcwd()}/topo/abilene_tm_edge.csv"
        for i in range(self.n):
            self.label_edge.append([])
            self.weights.append([])
            self.bw.append([])
            self.delay.append([])
            for j in range(self.n):
                self.label_edge[i].append('-')
                self.weights[i].append(int(0))
                self.bw[i].append(int(0))
                self.delay[i].append(int(0))
        with open(self.file_topo_edge) as csv_file:
            csv_node = csv.reader(csv_file, delimiter=' ')
            line_count = -1
            for row in csv_node:
                if line_count == -1:
                    line_count += 1
                else:
                    label = row[0]
                    src = int(row[1])
                    dest = int(row[2])
                    weight = int(row[3])
                    bw = int(row[4])
                    delay = int(row[5])
                    self.label_edge[src][dest] = label
                    self.weights[src][dest] = weight
                    self.bw[src][dest] = bw
                    self.delay[src][dest] = delay

        # init utilization
        for p in range(self.n):
            self.utilization.append([])
            for q in range(self.n):
                self.utilization[p].append(int(0))

        # init trafficMatrix
        for a in range(self.n):
            self.trafficMatrix.append([])
            for b in range(self.n):
                self.trafficMatrix[a].append(int(0))
        # init solution
        for aa in range(self.n):
            self.solution.append([])
            for bb in range(self.n):
                self.trafficMatrix[aa].append(int(0))

    def getReward(self):
        # self.trafficMatrix, self.solution, loadfile_utilization_step --> reward
        return 0

    def set_next_trafficMatrix(self):
        # load file
        # update self.trafficMatrix
        pass

    def set_next_utilization(self):
        # self.trafficMatrix, self.solution, --> next utilization
        # update self.utilization
        pass

if __name__ == "__main__":
    net = Network()
    net.loadTopo()
    state = net.getState()
    for obv in state:
        for i in range(net.getN()):
            for j in range(net.getN()):
                print(obv[i][j], end=" ")
            print()
        print()

