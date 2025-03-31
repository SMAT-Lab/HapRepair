let sync: displaySync.DisplaySync = displaySync.create();
sync.setExpectedFrameRateRange({
  expected: 60,
  min: 45,
  max: 60,
});