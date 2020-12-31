# -*- coding: utf-8 -*-
#############################################################

####################################################################################
import os
import re
import subprocess 
import time
import sched
import re
from time import sleep
import random
import datetime
import numpy as np
#import pyinotify
from numpy import array as matrix, arange
from dqn2 import DQN

schedule = sched.scheduler(time.time,time.sleep)




def write_register(arr):
    #start = time.time()
    start = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-4]
    p = subprocess.Popen('simple_switch_CLI --thrift-port 9090',shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True) 
    p.stdin.write('register_write threshold 0 '+str(arr))
    out,err = p.communicate()

    p1 = subprocess.Popen('simple_switch_CLI --thrift-port 9091',shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    p1.stdin.write('register_write threshold 0 '+str(arr))

    out,err = p1.communicate()
    #end = time.time()
    #print('time:',end-start,'s')
    #print(s+" "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4])
    print("["+str(arr)+","+str(10-arr)+"]"+" "+start+" "+datetime.datetime.now().strftime('%H:%M:%S.%f')[:-4])

    #p = subprocess.Popen('simple_switch_CLI',shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True) 
    #p.stdin.write('register_read threshold '+x)
    #print('register_read threshold '+x)
    #out_1,err_1 = p.communicate()
    #queue = re.findall('threshold\[\d\]\= \d*', out_1, re.M)
    #print ('register_read:'+str(queue[0]))  
    #end = time.time()
    #print('time:', end, 's')


class MyEventHandler():#pyinotify.ProcessEvent
    def process_IN_CLOSE_WRITE(self, event):
        fo = open("/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes.txt", "r")  
        for line in fo.readlines():
            if line.strip()=='':
                continue
            line = line.strip()
            mes = line.split(' ')
            #print(mes)
            agent = DQN()
            agent.load()
            action=agent.choose_action(matrix(mes),larger_greedy=1)
            write_register(action)
        fo.close() 
        

def run():
    #wm = pyinotify.WatchManager()
    #wm.add_watch('/home/myp4/P4/tutorials/exercises/ycx-project/message', pyinotify.ALL_EVENTS, rec=True)
    # event handler
    #eh = MyEventHandler()
 
    # notifier
    #notifier = pyinotify.Notifier(wm, eh)
    #notifier.loop()
    agent = DQN()
    agent.load()
    state=np.zeros((1,6))
    action=5
    f1 = "/home/myp4/P4/tutorials/exercises/ycx-project/message/newmes.txt"
    while True:
            fo = open("/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes.txt", "r")  
            for line in fo.readlines():
              if line.strip()=='':
                continue
              line = line.strip()
              #mes = line.split(' ')
              #print(mes)
            mes=np.loadtxt("/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes.txt")
            act=agent.choose_action(mes,larger_greedy=1)
            write_register(act)
            r=mes
            print(r)
            reward=(r[2]+r[5])-(r[1]+r[4])/100-(r[0]+r[3])/100
            agent.store_transition(state,action,reward,r)
            with open(f1,"a") as file:   
              #file.write(str(state)+" "+str(action)+" "+str(reward)+" "+str(r)+'\n')        
              file.write(str(act)+" "+str(line)+'\n')
            state=r
            action=act
            agent.learn()
            fo.close() 
            time.sleep(5)


if __name__ == "__main__":
    run()


