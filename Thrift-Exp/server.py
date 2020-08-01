import json
from sample import Transmit
from sample.ttypes import *
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import socket


class TransmitHandler:
	def __init__(self):
		self.log = {}

	def sayMsg(self, msg):
		msg = json.loads(msg)
		print("sayMsg(" + msg + ")")
		return "say " + msg + " from " + socket.gethostbyname(socket.gethostname())

	def invoke(self,cmd,token,data):
		cmd = cmd
		token =token
		data = data
		if cmd ==1:
			return json.dumps({token:data})
		else:
			return 'cmd不匹配'

if __name__=="__main__":
	handler = TransmitHandler()
	processor = Transmit.Processor(handler)
	transport = TSocket.TServerSocket('127.0.0.1', 8000)
	tfactory = TTransport.TBufferedTransportFactory()
	pfactory = TBinaryProtocol.TBinaryProtocolFactory()
	server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
	print("Starting python server...")
	server.serve()