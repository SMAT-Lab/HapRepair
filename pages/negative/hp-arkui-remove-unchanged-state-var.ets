// 建议移除未改变的状态变量设置。
//
// 通用丢帧场景下，建议优先修改。

@Observed
class Translate {
  translateX: number = 20;
}
@Component
struct Title {
  build() {
    Row() {
      // 本地资源 icon.png
      Image($r('app.media.icon'))
        .width(50)
        .height(50)
      Text("Title")
        .fontSize(20)
    }
  }
}
@Entry
@Component
struct MyComponent{
  @State translateObj: Translate = new Translate();
  @State button_msg: string = "i am button";

  build() {
    Column() {
      Title()
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)
      // 这里只是用了状态变量button_msg的值，没有任何写的操作
      Button(this.button_msg)
        .onClick(() => {
          animateTo({
            duration: 50
          },()=>{
            this.translateObj.translateX = (this.translateObj.translateX + 50) % 150
          })
        })
    }
    .translate({
      x: this.translateObj.translateX
    })
  }
}