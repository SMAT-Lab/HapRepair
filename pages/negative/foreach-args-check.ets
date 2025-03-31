// 建议在ForEach参数中设置keyGenerator。
//
// 滑动丢帧场景下，建议优先修改。

@Entry
@Component
struct ForeachTest {
  private data: string[] = ['1', '2', '3'];
  build() {
    RelativeContainer() {
      List() {
        // ForEach缺少第三个参数，告警
        ForEach(this.data, (item: string, index: number) => {
          ListItem() {
            Text(item);
          }
        })
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
    .width('100%')
  }
}