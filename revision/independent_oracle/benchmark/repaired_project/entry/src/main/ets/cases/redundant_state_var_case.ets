@Entry
@Component
struct AuditPanel {
  private auditTrail: string = '';

  appendAudit(message: string): void {
    this.auditTrail += message;
  }

  build() {
    Column() {
      Text('Audit controls')
      Button('Record')
    }
  }
}
