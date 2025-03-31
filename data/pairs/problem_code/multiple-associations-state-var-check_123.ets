@Observed
class BatteryStatus {
  isCharging: boolean = true;
}

@Entry
@Component
struct DemoI {
  @State battery: BatteryStatus = new BatteryStatus();

  build() {
    Column() {
      BatteryIcon({ battery: this.battery })
      BatteryInfo({ battery: this.battery })
      Text('Toggle Charging')
        .onClick(() => {
          this.battery.isCharging = !this.battery.isCharging;
        })
    }
  }
}

@Component
struct BatteryIcon {
  @Link battery: BatteryStatus;

  build() {
    Text('Battery Icon')
      .text(this.battery.isCharging ? 'Charging' : 'Discharging')
  }
}

@Component
struct BatteryInfo {
  @Link battery: BatteryStatus;

  build() {
    Text('Battery Info')
      .text(this.battery.isCharging ? 'Power Source Connected' : 'Running on Battery')
  }
}