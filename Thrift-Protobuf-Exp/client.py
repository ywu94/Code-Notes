import sys
import json
from sample import Transmit
from sample.ttypes import *
from sample.constants import *
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import sample_pb2

def get_connection():
	transport = TSocket.TSocket('127.0.0.1', 8000)
	transport = TTransport.TBufferedTransport(transport)
	protocol = TBinaryProtocol.TBinaryProtocol(transport)
	client = Transmit.Client(protocol)
	transport.open()
	return client, transport


if __name__=='__main__':
	cmd = 1
	token = '1111-2222-3333-4444'
	data = json.dumps({"name":"zhoujielun"})
	with open('address_book.pb', 'rb') as f:
		pbstring = f.read()

	client, transport = get_connection()

	# msg = client.sayMsg(json.dumps('lower case'))
	# msg = client.invoke(cmd,token,data)
	msg = client.readPB(pbstring)

	print(msg)

	transport.close()