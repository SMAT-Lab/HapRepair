let range : ExpectedFrameRateRange = {
  expected: 30,
  min: 0,
  max: 120
};

let draw30 = (intervalInfo: displaySync.IntervalInfo) => {
  if (this.isBigger_30) {
    this.drawFirstSize += 1;
    if (this.drawFirstSize > 150) {
      this.isBigger_30 = false;
    }
  } else {
    this.drawFirstSize -= 1;
    if (this.drawFirstSize < 25) {
      this.isBigger_30 = true;
    }
  }
};

this.backDisplaySyncSlow = displaySync.create();
this.backDisplaySyncSlow.setExpectedFrameRateRange(range);
this.backDisplaySyncSlow.on("frame", draw30);

