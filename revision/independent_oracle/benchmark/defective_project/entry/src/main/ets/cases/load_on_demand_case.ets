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

  build() {
    List({ space: 6 }) {
      ForEach(this.alerts, (alert: string) => {
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
