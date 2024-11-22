@Entry
@Component
struct ButtonComponent {
  @State buttons: string[] = ["Button 1", "Button 2", "Button 3", "Button 4", "Button 5"];

  build() {
    Column() {
      ForEach(this.buttons, (item: string) => {
        Column() {
          Column() {
            Column() {
              Button() {
                Text(item)
                  .fontSize(16)
                  .backgroundColor(Color.Green)
                  .padding(10)
                  .onClick(() => {
                    console.log(`${item} clicked`);
                  })
              }
              .height(50)
            }
            .width("100%")
          }
          .padding(5)
        }
        .margin(5)
      }, (item: string) => item)
    }
  }
}
