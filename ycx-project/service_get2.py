# -*- coding: utf-8 -*-
#############################################################
#server.py
import pickle
from multiprocessing.connection import Listener
from threading import Thread
import commands


class RPCHandler:
    def __init__(self):
        self._functions = {}

    def register_function(self, func):
        self._functions[func.__name__] = func

    def handle_connection(self, connection):
        try:
            while True:
                # Receive a message
                func_name, args, kwargs = pickle.loads(connection.recv())
                # Run the RPC and send a response
                try:
                    r = self._functions[func_name](*args, **kwargs)
                    connection.send(pickle.dumps(r))
                except Exception as e:
                    connection.send(pickle.dumps(e))
        except EOFError:
            pass


def rpc_server(handler, address, authkey):
    sock = Listener(address, authkey=authkey)
    while True:
        client = sock.accept()
        t = Thread(target=handler.handle_connection, args=(client,))
        t.daemon = True
        t.start()


# Some remote functions
def get_mes2():
    out = commands.getstatusoutput('ping -i 0.2 -c 25 10.0.2.6')
    time = out[1].split('/')[4]
    #print(out)
    drop = out[1].split('%')[0].split(' ')[-1]
    return time+" "+drop
   


# Register with a handler
handler = RPCHandler()
handler.register_function(get_mes2)

# Run the server
rpc_server(handler, ('10.0.1.5', 17000), authkey=b'boyun2')
