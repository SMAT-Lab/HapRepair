@Entry
@Component
struct BatchReporter {
  @State processedCount: number = 0;

  build() {
    Column() {
      Text(`Processed: ${this.processedCount}`)
      Button('Process batch')
        .onClick(() => {
          for (let index: number = 0; index < 8; index++) {
            this.processedCount += 1;
          }
        })
    }
  }
}
