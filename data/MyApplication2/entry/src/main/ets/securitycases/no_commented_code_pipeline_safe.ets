@Component
struct PipelineMonitorClean {
  @State batches: Array<string> = [];

  record(batch: string): void {
    this.batches.push(batch);
  }

  build() {
    Column() {
      Text(`count: ${this.batches.length}`)
      Button('Push Batch')
        .onClick(() => {
          this.record(`batch-${this.batches.length}`);
        })
    }
  }
}
