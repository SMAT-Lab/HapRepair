@Component
struct SensorWidget {
  @State temperature: number = 25;

  refreshMetrics(): void {
    this.temperature += 1;
    // this.temperature = sensor.read();
  }

  build() {
    Column() {
      Text(`Temp: ${this.temperature}`)
      Button('Calibrate')
        .onClick(() => {
          this.refreshMetrics();
          // this.temperature -= 5;
        })
    }
  }
}
