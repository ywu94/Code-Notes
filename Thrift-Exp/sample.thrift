service Transmit {
	string sayMsg(1:string msg);
	string invoke(1:i32 cmd 2:string token 3:string data)
}