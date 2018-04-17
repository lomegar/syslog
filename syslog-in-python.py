# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:57:15 2018

@author: dragonv
"""

#!/usr/bin/python

## Tiny Syslog Server in Python.
##
## This is a tiny syslog server that is able to receive UDP based syslog
## entries on a specified port and save them in Redis
## Org file at https://gist.github.com/marcelom/4218010
##

import socketserver
import time
import redis
import pandas as pd
import re

r = redis.StrictRedis(host='localhost', port=6379, db=0)
pipe = r.pipeline()
HOST, PORT = "0.0.0.0", 514
lineNumber = 1
count = 1


class SyslogUDPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global lineNumber        
        global count                     
        data = bytes.decode(self.request[0].strip(), 'utf-8')
        pipe.set(lineNumber,data)
        pipe.execute()            
        lineNumber += 1
        
        if lineNumber > 10000*count:
            logdata = pd.DataFrame()  
            for key in r.scan_iter():
                print('keys is%s'%key)
                lograw = pipe.get(key)
                pipe.delete(key)
                pipe.execute()
                print('log is %s'%lograw)
                loglist = re.split(r'\s|\|',lograw)
                print(loglist)                
                logdata = logdata.append(loglist,ignore_index=True)
                logdata = logdata
            logdata.to_csv('\data\syslog\%s'+'.csv'%int(time.time()))
            count += 1
            
        if lineNumber > 50000000:
            lineNumber = 1
            count = 1
            
if __name__ == "__main__":
    try:
        server = socketserver.UDPServer((HOST, PORT), SyslogUDPHandler)
        server.serve_forever(poll_interval=0.5)
    except (IOError, SystemExit):
        pass
    except KeyboardInterrupt:
        print ("Crtl+C Pressed. Shutting down.")