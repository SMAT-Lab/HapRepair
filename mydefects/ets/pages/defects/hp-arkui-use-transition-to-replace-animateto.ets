@Entry
@Component
struct MyComponent {
  @State mOpacity: number = 1;
  @State visible: boolean = true;

  build() {
    Column() {
      Row() {
        if (this.visible) {
          Text('Component Content')
            .opacity(this.mOpacity)
        }
      }
      .width('100%')
      .height(100)
      .justifyContent(FlexAlign.Center)

      Button('Toggle Visibility')
        .onClick(() => {
          this.visible = true;
          animateTo({
            duration: 800,
            onFinish: () => {
              if (this.mOpacity === 0) {
                this.visible = false;
              }
            }
          }, () => {
            this.mOpacity = this.mOpacity === 1 ? 0 : 1;
          });
        })
        .fontSize(16)
        .padding(10)
        .backgroundColor(Color.Blue)
    }
  }
}