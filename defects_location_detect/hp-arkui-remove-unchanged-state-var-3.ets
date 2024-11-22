@Entry
@Component
struct MyComponent {
  @State userName: string = "Guest";
  @State userStatus: string = "Active";

  build() {
    Column() {
      Text("User Name: " + this.userName)
        .fontSize(16)
        .fontColor(Color.Black)
        .padding(10)

      Text("User Status: " + this.userStatus)
        .fontSize(16)
        .fontColor(Color.Black)
        .padding(10)

      Button("Change Status")
        .onClick(() => {
          console.log("Button clicked");
        })
        .fontSize(16)
        .backgroundColor(Color.Green)
        .padding(10)
    }
    .width('100%')
    .height('100%')
  }
}
