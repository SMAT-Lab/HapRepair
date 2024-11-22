@Entry
@Component
struct ChatComponent {
  @State messages: string[] = Array.from(Array(1000), (_, i) => `Message ${i + 1}`);

  build() {
    Column() {
      Text('聊天记录')
        .fontSize(24)
        .padding(10)
        .textAlign(TextAlign.Center)

      // 聊天记录列表
      List() {
        LazyForEach(this.messages, (message: string) => {
          ListItem() {
            // 使用 ChatMessage 自定义组件，嵌套显示每条聊天记录
            ChatMessage({ messageContent: message })
          }
        }, (message: string) => message)
      }
      .height(500)
      .width('100%')
      .backgroundColor(Color.White)
    }
  }
}

// 定义的聊天消息组件，用于显示单条聊天记录
@Reusable
@Component
struct ChatMessage {
  @State messageContent: string = '';

  aboutToReuse(params: ESObject): void {
    this.messageContent = params.messageContent as string;
  }

  build() {
    Row() {
      Text(this.messageContent)
        .fontSize(18)
        .padding(10)
    }
    .backgroundColor(0xF1F1F1)
    .padding(5)
  }
}
