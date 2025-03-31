// 不允许锁定最高帧率运行。

let sync = displaySync.create();
sync.setExpectedFrameRateRange({
  expected: 120,
  min: 120,
  max: 120,
});