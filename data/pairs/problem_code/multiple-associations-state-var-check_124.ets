@Observed
class NetworkConfig {
  isConnected: boolean = true;
}

@Entry
@Component
struct DemoJ {
  @State network: NetworkConfig = new NetworkConfig();

  build() {
    Row() {
      StatusIndicator({ network: this.network })
      DataUsage({ network: this.network })
      Text('Connect/Disconnect')
        .onClick(() => {
          this.network.isConnected = !this.network.isConnected;
        })
    }
  }
}

@Component
struct StatusIndicator {
  @Link network: NetworkConfig;

  build() {
    Text('Status Indicator')
      .text(this.network.isConnected ? 'Connected' : 'Disconnected')
  }
}

@Component
struct DataUsage {
  @Link network: NetworkConfig;

  build() {
    Text('Data Usage')
      .text(this.network.isConnected ? 'Tracking Usage' : 'Offline Mode')
  }
}