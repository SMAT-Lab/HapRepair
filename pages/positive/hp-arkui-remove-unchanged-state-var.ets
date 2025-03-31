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
  // 直接使用一般变量即可
  button_msg: string = "i am button";

  build() {
    Column() {
      Title()
      Stack() {
      }
      .backgroundColor("black")
      .width(200)
      .height(400)

      Button(this.button_msg)
        .onClick(() => {
          animateTo({
            duration: 50
          }, () => {
            this.translateObj.translateX = (this.translateObj.translateX + 50) % 150
          })
        })
    }
    .translate({
      x: this.translateObj.translateX
    })
  }
}