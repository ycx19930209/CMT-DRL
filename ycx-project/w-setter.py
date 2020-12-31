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
from numpy import array as matrix, arange

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

def run():
    f1 = "/home/myp4/P4/tutorials/exercises/ycx-project/message/ranmes.txt"
 
    for i in range(2000):
        #read states
        fo = open("/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes.txt", "r")  
        for line in fo.readlines():
            if line.strip()=='':
                continue
            line = line.strip()
            mes = line.split(' ')
        
        #b = random.sample(range(0,10),1)
        b = random.randint(0,10)
        write_register(b)
        #####record action t and state t
        with open(f1,"a") as file:   
          file.write(str(b)+" "+str(line)+'\n')
        fo.close()
        time.sleep(5)
     

if __name__ == "__main__":
    run()


