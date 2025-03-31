
  @State message: string = '';
  appendMsg(newMsg: String) {
    this.message += newMsg;
    this.message += ";";
    this.message += "<br/>";
  }