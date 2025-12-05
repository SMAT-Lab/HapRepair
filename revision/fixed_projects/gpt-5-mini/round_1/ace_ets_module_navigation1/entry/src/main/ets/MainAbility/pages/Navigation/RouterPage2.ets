
@Entry
@Component
struct RouterPage2 {
  // 直接使用一般变量即可
  message: string = 'routerPage2';

  build() {
    Column({space: '20vp'}) {
      Text(this.message)
        .id('RouterPage2Id')
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
    }.margin({top: '20vp'})
    .height('100%')
    .width('100%')
  }
}