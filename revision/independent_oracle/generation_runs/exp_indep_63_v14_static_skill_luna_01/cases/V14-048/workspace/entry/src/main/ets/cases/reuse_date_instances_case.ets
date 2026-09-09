class ClockSampler {
  private displayValue: string = '';
  private current: Date = new Date();

  start(): void {
    setInterval(() => {
      this.current.setTime(Date.now());
      this.displayValue = this.current.toISOString();
    }, 1000);
  }
}
