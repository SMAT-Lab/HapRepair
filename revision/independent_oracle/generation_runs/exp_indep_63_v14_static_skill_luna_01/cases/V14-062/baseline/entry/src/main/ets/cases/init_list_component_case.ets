@Component
struct CompactMessageList {
  private messages: string[] = ['Queued', 'Running', 'Done'];

  build() {
    List() {
      ForEach(this.messages, (message: string) => {
        ListItem() {
          Text(message)
        }
      }, (message: string) => message)
    }
    .width('92%')
  }
}
