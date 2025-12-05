@Entry
@Component
struct SyncPanelClean {
  @State status: string = 'idle';

  startSync(): void {
    this.status = 'running';
  }

  build() {
    Column() {
      Text(`status: ${this.status}`)
      Button('Start')
        .onClick(() => {
          this.startSync();
        })
    }
  }
}
