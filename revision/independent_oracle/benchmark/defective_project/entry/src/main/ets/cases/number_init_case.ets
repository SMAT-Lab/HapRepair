function revisedThreshold(enabled: boolean): number {
  let threshold = 4;
  if (enabled) {
    threshold = 4.75;
  }
  return threshold;
}
