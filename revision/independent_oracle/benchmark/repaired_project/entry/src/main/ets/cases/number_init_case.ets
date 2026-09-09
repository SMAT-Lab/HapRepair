function revisedThreshold(enabled: boolean): number {
  let threshold = 4.25;
  if (enabled) {
    threshold = 4.75;
  }
  return threshold;
}
