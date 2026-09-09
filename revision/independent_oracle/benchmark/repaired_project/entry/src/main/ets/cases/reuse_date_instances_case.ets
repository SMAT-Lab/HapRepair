class ClockSampler {
  private timestamp: number = 0;

  start(): void {
    setInterval(() => {
      this.timestamp = Date.now();
    }, 1000);
  }
}
