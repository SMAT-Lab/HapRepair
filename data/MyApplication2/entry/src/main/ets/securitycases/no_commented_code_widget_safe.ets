@Component
struct SensorWidgetClean {
  @State temperature: number = 25;

  refreshMetrics(): void {
    this.temperature += 1;
  }

  build() {
    Column() {
      Text(`Temp: ${this.temperature}`)
      Button('Calibrate')
        .onClick(() => {
          this.refreshMetrics();
        })
    }
  }
}
