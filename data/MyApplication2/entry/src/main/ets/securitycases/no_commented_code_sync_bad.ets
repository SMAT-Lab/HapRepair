@Entry
@Component
struct SyncPanel {
  @State status: string = 'idle';

  startSync(): void {
    this.status = 'running';
    // network.fetch('/sync/all');
  }

  build() {
    Column() {
      Text(`status: ${this.status}`)
      Button('Start')
        .onClick(() => {
          this.startSync();
          // this.status = 'failed';
        })
    }
  }
}
