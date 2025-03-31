@Entry
@Component
struct App {
  build() {
    VStack() {
      ForEach(this.messages, (msg, index) => {
        MessageBubble(msg);
      }, (msg) => msg.uuid);
    }
  }
}