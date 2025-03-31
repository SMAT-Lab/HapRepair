let range : ExpectedFrameRateRange = {
  expected: 120,
  min: 120,
  max: 120
};
let backDisplaySyncSlow: displaySync.DisplaySync;
backDisplaySyncSlow = displaySync.create();
backDisplaySyncSlow.setExpectedFrameRateRange(range);