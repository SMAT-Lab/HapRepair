class AlertDataSource implements IDataSource {
  private items: string[]
  private listeners: DataChangeListener[] = []

  constructor(items: string[]) {
    this.items = items
  }

  totalCount(): number {
    return this.items.length
  }

  getData(index: number): string {
    return this.items[index]
  }

  registerDataChangeListener(listener: DataChangeListener): void {
    if (this.listeners.indexOf(listener) < 0) {
      this.listeners.push(listener)
    }
  }

  unregisterDataChangeListener(listener: DataChangeListener): void {
    const index: number = this.listeners.indexOf(listener)
    if (index >= 0) {
      this.listeners.splice(index, 1)
    }
  }
}

@Entry
@Component
struct ServiceAlertInbox {
  private alerts: string[] = [
    'payment-latency',
    'search-timeout',
    'image-backlog',
    'quota-warning',
    'sync-delayed',
    'cache-pressure',
    'auth-retry',
    'storage-near-limit',
    'worker-stalled',
    'network-jitter',
    'upload-paused',
    'download-slow',
    'index-rebuilding',
    'replica-lag',
    'queue-depth-high',
    'certificate-expiring',
    'clock-drift',
    'memory-pressure',
    'cpu-saturation',
    'service-degraded'
  ];
  private alertDataSource: AlertDataSource = new AlertDataSource(this.alerts)

  build() {
    List({ space: 6 }) {
      LazyForEach(this.alertDataSource, (alert: string) => {
        ListItem() {
          Row({ space: 8 }) {
            Text('!').fontColor(Color.Red)
            Text(alert)
          }
        }
      }, (alert: string) => alert)
    }
    .width('92%')
    .height(360)
  }
}
