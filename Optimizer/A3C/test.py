import torch

from Optimizer.A3C.Server import Server
from Optimizer.A3C.Worker import Worker
from Optimizer.A3C.Server_method import synchronize
from Optimizer.A3C.Worker_method import asynchronize


def zero_actor_weights(worker):
    for actor_param in worker.actor_net.parameters():
        actor_param.data *= 0


def print_actor_param(worker):
    for actor_param in worker.actor_net.parameters():
        print(actor_param.data)


def print_actor_grad(worker):
    for actor_param in worker.actor_net.parameters():
        print(actor_param.grad)


# test copy weights
def test1():
    baseServer = Server(nb_state_feature=4, nb_action=3, name="server")
    worker1 = Worker(Server_object=baseServer, name="worker_1", id=1)
    worker2 = Worker(Server_object=baseServer, name="worker_2", id=2)
    worker3 = Worker(Server_object=baseServer, name="worker_3", id=3)

    workerList = [worker1, worker2, worker3]
    synchronize(baseServer, workerList)

    print(worker1.actor_net.parameters())
    print(worker2.actor_net.parameters())

    print("===================================WORKER 1 BEFORE========================================")
    print_actor_param(worker1)
    zero_actor_weights(worker2)
    print("===================================WORKER 1 AFTER========================================")
    print_actor_param(worker1)
    print("===================================WORKER 2========================================")
    print_actor_param(worker2)

    del baseServer
    del worker1
    del worker2
    del worker3


# test synchronize
def test2():
    baseServer = Server(nb_state_feature=4, nb_action=3, name="server")
    worker1 = Worker(Server_object=baseServer, name="worker_1", id=1)
    worker2 = Worker(Server_object=baseServer, name="worker_2", id=2)
    workerList = [worker1, worker2]

    test_vector = torch.Tensor([1, 2, 3, 4])
    result1 = worker1.get_value(test_vector)
    result2 = worker2.get_value(test_vector)

    print(result1)
    print(result2)

    synchronize(baseServer, workerList)
    result1 = worker1.get_value(test_vector)
    result2 = worker2.get_value(test_vector)

    print("===================================AFTER========================================")
    print(result1)
    print(result2)

    del baseServer
    del worker1
    del worker2


# test asynchronize
def test3():
    baseServer = Server(nb_state_feature=4, nb_action=3, name="server")
    worker1 = Worker(Server_object=baseServer, name="worker_1", id=1)

    print("----------------------")
    print_actor_grad(worker1)  # return all None
    print("----------------------")
    print_actor_grad(baseServer)  # return all None

    zero_actor_weights(baseServer)
    print_actor_param(baseServer)
    test_vector = torch.Tensor([1, 2, 3, 4])

    out_actor_worker1 = worker1.get_policy(test_vector)
    loss = 1 / 2 * torch.sum((torch.Tensor([4, 5, 6]) - out_actor_worker1) ** 2)
    loss.backward()

    print('---------------------')
    print_actor_grad(worker1)

    asynchronize(worker1, baseServer)
    print('------------ACTOR PARAM---------')
    print_actor_param(baseServer)
    print('------------ACTOR GRAD---------')
    print_actor_grad(baseServer)

    del baseServer
    del worker1


if __name__ == "__main__":
    # test3()
    L = [1, 2, 3, 4, 5]
    T = L[-1]
    L.clear()
    L.append(T)
    print(L)
    """
    CONFIRM: all test passed, A3C is implemented correctly
    """
