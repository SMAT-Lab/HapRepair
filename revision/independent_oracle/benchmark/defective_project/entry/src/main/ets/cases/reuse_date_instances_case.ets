class ClockSampler {
  private displayValue: string = '';

  start(): void {
    setInterval(() => {
      const current: Date = new Date();
      this.displayValue = current.toISOString();
    }, 1000);
  }
}
