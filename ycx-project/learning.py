# coding=utf-8

from numpy import array as matrix, arange
from dqn2 import DQN
# group_config = VNFGroupConfig()

if __name__ == '__main__':
    agent = DQN()
    agent.load()
    fo = open("/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes.txt", "r")  
    for line in fo.readlines():
        if line.strip()=='':
            continue
        line = line.strip()
        mes = line.split(' ') 
    fo.close()
    action=agent.choose_action(matrix(mes),larger_greedy=1)
    print(action)


