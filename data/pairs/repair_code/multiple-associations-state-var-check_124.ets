@Observed
class NetworkConfig {
  isConnected: boolean = true;
}

@Entry
@Component
struct DemoJFixed {
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
  @Link @Watch('onConnectionChange') network: NetworkConfig;
  @State localIsConnected: boolean = true;

  onConnectionChange() {
    this.localIsConnected = this.network.isConnected;
  }

  build() {
    Text('Status Indicator')
      .text(this.localIsConnected ? 'Connected' : 'Disconnected')
  }
}

@Component
struct DataUsage {
  @Link @Watch('onConnectionChange') network: NetworkConfig;
  @State localIsConnected: boolean = true;

  onConnectionChange() {
    this.localIsConnected = this.network.isConnected;
  }

  build() {
    Text('Data Usage')
      .text(this.localIsConnected ? 'Tracking Usage' : 'Offline Mode')
  }
}