@Entry
@Component
struct StatusToolbar {
  build() {
    Flex({ direction: FlexDirection.Row }) {
      Flex() { Text('Queued') }.padding(6).backgroundColor(0xFFF3CD)
      Flex() { Text('Running') }.padding(6).backgroundColor(0xD1ECF1)
      Flex() { Text('Blocked') }.padding(6).backgroundColor(0xF8D7DA)
      Flex() { Text('Review') }.padding(6).backgroundColor(0xE2E3E5)
      Flex() { Text('Passed') }.padding(6).backgroundColor(0xD4EDDA)
      Flex() { Text('Archived') }.padding(6).backgroundColor(0xE8DAEF)
    }
    .width('100%')
    .height(64)
  }
}
