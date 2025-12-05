@Entry
@Component
struct AuditBoard {
  @State logs: Array<string> = [];

  addRecord(record: string): void {
    this.logs.push(`${new Date().toUTCString()}: ${record}`);
    // hilog.info(1000, 'AuditBoard', record);
  }

  build() {
    Column() {
      Text(`records: ${this.logs.length}`)
        .fontSize(18)
      Button('Sync Ledger')
        .onClick(() => {
          this.addRecord('sync started');
          // this.logs = [];
        })
    }
  }
}
