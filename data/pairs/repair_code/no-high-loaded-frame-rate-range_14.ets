let range : ExpectedFrameRateRange = {
  expected: 30,
  min: 0,
  max: 120
};
let backDisplaySyncSlow: displaySync.DisplaySync;
backDisplaySyncSlow = displaySync.create();
backDisplaySyncSlow.setExpectedFrameRateRange(range);