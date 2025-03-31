function setupSync() {
  const frameSync = displaySync.create();
  frameSync.configureRateRange({
    expected: 144,
    min: 144,
    max: 144
  });
}