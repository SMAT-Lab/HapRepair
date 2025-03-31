@Reusable
@Component
struct MessageItem {
  @State sender: string = '';
  @State content: string = '';

  aboutToReuse(params: Record<string, Object>): void {
    this.sender = params.sender as string;
    this.content = params.content as string;
  }

  build() {
    Column() {
      Text('发件人：' + this.sender)
      Text('内容：' + this.content)
    }
  }
}

interface GeneratedTypeLiteralInterface_1 {
  sender: string;
  content: string;
}

@Entry
@Component
struct MessageList {
  private messages: Array<GeneratedTypeLiteralInterface_1> = [];

  aboutToAppear(): void {
    // 获取消息列表
    this.messages = [
      { sender: '张三', content: '你好' },
      { sender: '李四', content: '下午好' },
    ];
  }

  build() {
    Column() {
      List() {
        ForEach(this.messages, (message: GeneratedTypeLiteralInterface_1) => {
          ListItem() {
            MessageItem({ sender: message.sender, content: message.content })
          }
        }, (message:GeneratedTypeLiteralInterface_1) => message.content)
      }
      .width("100%")
      .height("100%")
    }
  }
}
