# -*- coding: utf-8 -*-
#############################################################
#client.py
import pickle
from multiprocessing.connection import Client
import datetime
import thread
import time
class RPCProxy:
    def __init__(self, connection):
        self._connection = connection

    def __getattr__(self, name):
        def do_rpc(*args, **kwargs):
            self._connection.send(pickle.dumps((name, args, kwargs)))
            result = pickle.loads(self._connection.recv())
            if isinstance(result, Exception):
                raise result
            return result

        return do_rpc


c = Client(('10.0.1.3', 17000), authkey=b'boyun1')
proxy1 = RPCProxy(c)
d = Client(('10.0.1.5', 17000), authkey=b'boyun2')
proxy2 = RPCProxy(d)
path1=''
path2=''
f = "/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes.txt"
f1 = "/home/myp4/P4/tutorials/exercises/ycx-project/message/pathmes1.txt"
def getpath1():
    global path1
    path1=proxy1.get_mes1()
def getpath2():
    global path2
    path2=proxy2.get_mes2()
while True:
    try:
        thread.start_new_thread(getpath1, () )
        thread.start_new_thread(getpath2, () )
    except:
        print "Error: unable to start thread"
    time.sleep(5)
    mes = path1+" 1 "+path2+" 2"
    with open(f,"w") as file:   
        file.write(mes)
    with open(f1,"a") as file:   
        file.write(mes+'\n')
