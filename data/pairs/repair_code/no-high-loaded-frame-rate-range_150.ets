function setupSync() {
  const frameSync = displaySync.create();
  frameSync.configureRateRange({
    expected: 60,
    min: 30,
    max: 60
  });
}