  @State message: string = "";
  appendMsg(newMsg: String) : string {
    this.message += newMsg;
    return this.message;
  }