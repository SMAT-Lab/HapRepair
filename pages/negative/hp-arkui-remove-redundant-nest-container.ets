// 避免冗余的嵌套。
//
// 通用丢帧场景下，建议优先修改。

@Entry
@Component
struct MyComponent {
  @State children: Number[] = Array.from(Array<number>(900), (v, k) => k);

  build() {
    Scroll() {
      Grid() {
        ForEach(this.children, (item: Number[]) => {
          GridItem() {
            // 冗余Stack
            Stack() {
              Stack() {
                Stack() {
                  Text(item.toString())
                }.size({ width: "100%"})
              }.backgroundColor(Color.Yellow)
            }.backgroundColor(Color.Pink)
          }
        }, (item: string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr 1fr')
      .columnsGap(0)
      .rowsGap(0)
      .size({ width: "100%", height: "100%" })
    }
  }
}