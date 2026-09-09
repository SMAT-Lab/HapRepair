@Entry
@Component
struct BatchReporter {
  @State processedCount: number = 0;

  build() {
    Column() {
      Text(`Processed: ${this.processedCount}`)
      Button('Process batch')
        .onClick(() => {
          let count: number = this.processedCount;
          for (let index: number = 0; index < 8; index++) {
            count += 1;
          }
          this.processedCount = count;
        })
    }
  }
}
