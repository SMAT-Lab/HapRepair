@Observed
class BatteryStatus {
  isCharging: boolean = true;
}

@Entry
@Component
struct DemoIFixed {
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
  @Link @Watch('onChargeChange') battery: BatteryStatus;
  @State localIsCharging: boolean = true;

  onChargeChange() {
    this.localIsCharging = this.battery.isCharging;
  }

  build() {
    Text('Battery Icon')
      .text(this.localIsCharging ? 'Charging' : 'Discharging')
  }
}

@Component
struct BatteryInfo {
  @Link @Watch('onChargeChange') battery: BatteryStatus;
  @State localIsCharging: boolean = true;

  onChargeChange() {
    this.localIsCharging = this.battery.isCharging;
  }

  build() {
    Text('Battery Info')
      .text(this.localIsCharging ? 'Power Source Connected' : 'Running on Battery')
  }
}