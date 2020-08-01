import sys
import json
from sample import Transmit
from sample.ttypes import *
from sample.constants import *
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


transport = TSocket.TSocket('127.0.0.1', 8000)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = Transmit.Client(protocol)
# Connect!
transport.open()

cmd = 1
token = '1111-2222-3333-4444'
data = json.dumps({"name":"zhoujielun"})
# msg = client.invoke(cmd,token,data)
msg = client.sayMsg(json.dumps('lower case'))
print(msg)
transport.close()