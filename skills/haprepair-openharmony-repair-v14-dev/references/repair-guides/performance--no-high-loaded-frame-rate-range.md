# @performance/no-high-loaded-frame-rate-range

Static repair references: 15. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0855acdbb3f5787a`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
let sync: displaySync.DisplaySync = displaySync.create();
sync.setExpectedFrameRateRange({
  expected: 120,
  min: 120,
  max: 120,
});
```

### Repair pattern

```arkts
let sync: displaySync.DisplaySync = displaySync.create();
sync.setExpectedFrameRateRange({
  expected: 60,
  min: 45,
  max: 60,
});
```

### Rationale

在给定的代码中，通过调用 setExpectedFrameRateRange 方法，将帧率锁定在 120 帧每秒 (FPS)，会对设备的 CPU 和 GPU 造成很大的性能负担，尤其是在移动设备上。通过将帧率范围设置为 45-60 FPS，可以确保应用在大多数设备上都能稳定运行，同时不会对设备造成不必要的性能负担。这样可以平衡性能和电池消耗，同时确保用户体验的一致性。

## Example 2: `pair_1242ce7c51c48390`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
Slider({ value: this.curTime, min: 0, max: 100 })
  .enabled(false)
  .height(4)
  .width(this.xComponentWidth)
  .trackThickness(3)
  .blockColor(Color.Red)
  .blockSize({ width: 4, height: 4 })
  .onVisibleAreaChange([0.0, 1.0], (isVisible: boolean, currentRatio: number) => {
    if (isVisible && currentRatio >= 1.0) {
      animateTo({
        duration: 30000,
        iterations: -1,
        expectedFrameRateRange: {
          expected: 120,
          min: 120,
          max: 120,
        },
      }, () => {
        if (this.curTime >= 100) {
          this.curTime = 0;
        }
        for (let i = 0; i < 101; i++) {
          this.curTime += 1;
        }
      })
    }
  })
```

### Repair pattern

```arkts
Slider({ value: this.curTime, min: 0, max: 100 })
  .enabled(false)
  .height(4)
  .width(this.xComponentWidth)
  .trackThickness(3)
  .blockColor(Color.Red)
  .blockSize({ width: 4, height: 4 })
  .onVisibleAreaChange([0.0, 1.0], (isVisible: boolean, currentRatio: number) => {
    if (isVisible && currentRatio >= 1.0) {
      animateTo({
        duration: 30000,
        iterations: -1,
        expectedFrameRateRange: {
          expected: 30,
          min: 0,
          max: 120,
        },
      }, () => {
        if (this.curTime >= 100) {
          this.curTime = 0;
        }
        for (let i = 0; i < 101; i++) {
          this.curTime += 1;
        }
      })
    }
  })
```

### Rationale

在给定的代码中，通过设置expectedFrameRateRange ，将帧率锁定在 120 帧每秒 (FPS)，会对设备的 CPU 和 GPU 造成很大的性能负担，尤其是在移动设备上。通过将帧率范围设置为 0-120 FPS，可以确保应用在大多数设备上都能稳定运行，同时不会对设备造成不必要的性能负担。这样可以平衡性能和电池消耗，同时确保用户体验的一致性。

## Example 3: `pair_2760bb219ef8c391`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
function setupSync() {
  const frameSync = displaySync.create();
  frameSync.configureRateRange({
    expected: 144,
    min: 144,
    max: 144
  });
}
```

### Repair pattern

```arkts
function setupSync() {
  const frameSync = displaySync.create();
  frameSync.configureRateRange({
    expected: 60,
    min: 30,
    max: 60
  });
}
```

### Rationale

代码中配置了帧率范围，将其锁定在 144 帧每秒（FPS），给设备的 CPU 和 GPU 带来很大的性能负担。降低帧率范围可以缓解设备负担。

## Example 4: `pair_32cd043b6927d12b`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
@Widget
class FrameWidget {
  configure() {
    setRates({
      expected: 120,
      min: 120,
      max: 120
    });
  }
}
```

### Repair pattern

```arkts
@Widget
class FrameWidget {
  configure() {
    setRates({
      expected: 60,
      min: 30,
      max: 60
    });
  }
}
```

### Rationale

将帧率锁定在 120 FPS，会让设备面临较大的资源消耗，影响其他进程。

## Example 5: `pair_46dd1b4a6e09e447`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
function initializeFrame() {
  let syncConfig = {
    expected: 180,
    min: 180,
    max: 180
  };
  frameManager.setSync(syncConfig);
}
```

### Repair pattern

```arkts
function initializeFrame() {
  let syncConfig = {
    expected: 60,
    min: 30,
    max: 60
  };
  frameManager.setSync(syncConfig);
}
```

### Rationale

代码中将帧率锁定在 180 FPS，高负载可能会导致设备过热或电池消耗加快。

## Example 6: `pair_526c543457798c79`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
const settings = new FrameSettings(160, 160, 160);
frameManager.applySettings(settings);
```

### Repair pattern

```arkts
const settings = new FrameSettings(60, 30, 60);
frameManager.applySettings(settings);
```

### Rationale

帧率设置为 160 FPS，在很多设备上都会导致资源负载过重。

## Example 7: `pair_760a2b533c34e2fb`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
var config = {
  frameRate: {
    min: 200,
    expected: 200,
    max: 200
  }
};
display.setConfig(config);
```

### Repair pattern

```arkts
var config = {
  frameRate: {
    min: 30,
    expected: 60,
    max: 60
  }
};
display.setConfig(config);
```

### Rationale

此配置使帧率高达 200 FPS，硬件压力增大，可能会影响设备的整体性能表现。

## Example 8: `pair_89826e7d55ea6bae`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
let range : ExpectedFrameRateRange = {
  expected: 120,
  min: 120,
  max: 120
}

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
```

### Repair pattern

```arkts
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
```

### Rationale

在给定的代码中，通过设置expectedFrameRateRange ，将帧率锁定在 120 帧每秒 (FPS)，会对设备的 CPU 和 GPU 造成很大的性能负担，尤其是在移动设备上。通过将帧率范围设置为 0-120 FPS，可以确保应用在大多数设备上都能稳定运行，同时不会对设备造成不必要的性能负担。这样可以平衡性能和电池消耗，同时确保用户体验的一致性。

## Example 9: `pair_a1edecafa96b651e`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
@Entry
@Component
struct Index {
  build() {
    Button()
      .onClick(() => {
        animateTo({
          duration: 1200,
          iterations: 10,
          expectedFrameRateRange: { 
            expected: 120,
            min: 120,
            max: 120,
          },
        }, () => {
        })
      })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct Index {
  build() {
    Button()
      .onClick(() => {
        animateTo({
          duration: 1200,
          iterations: 10,
          expectedFrameRateRange: { 
            expected: 30,
            min: 0,
            max: 120,
          },
        }, () => {
        })
      })
  }
}
```

### Rationale

在给定的代码中，通过设置expectedFrameRateRange ，将帧率锁定在 120 帧每秒 (FPS)，会对设备的 CPU 和 GPU 造成很大的性能负担，尤其是在移动设备上。通过将帧率范围设置为 0-120 FPS，可以确保应用在大多数设备上都能稳定运行，同时不会对设备造成不必要的性能负担。这样可以平衡性能和电池消耗，同时确保用户体验的一致性。

## Example 10: `pair_a672ccf69f4af5d3`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
let range : ExpectedFrameRateRange = {
  expected: 120,
  min: 120,
  max: 120
};
let backDisplaySyncSlow: displaySync.DisplaySync;
backDisplaySyncSlow = displaySync.create();
backDisplaySyncSlow.setExpectedFrameRateRange(range);
```

### Repair pattern

```arkts
let range : ExpectedFrameRateRange = {
  expected: 30,
  min: 0,
  max: 120
};
let backDisplaySyncSlow: displaySync.DisplaySync;
backDisplaySyncSlow = displaySync.create();
backDisplaySyncSlow.setExpectedFrameRateRange(range);
```

### Rationale

在给定的代码中，通过调用 setExpectedFrameRateRange 方法，将帧率锁定在 120 帧每秒 (FPS)，会对设备的 CPU 和 GPU 造成很大的性能负担，尤其是在移动设备上。通过将帧率范围设置为 0-30 FPS，可以确保应用在大多数设备上都能稳定运行，同时不会对设备造成不必要的性能负担。这样可以平衡性能和电池消耗，同时确保用户体验的一致性。

## Example 11: `pair_a98553283232568f`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
@Entry
@Component
struct AnimationComponent {
  setup() {
    animate({
      duration: 5000,
      frameRate: {
        expected: 144,
        min: 144,
        max: 144
      }
    });
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct AnimationComponent {
  setup() {
    animate({
      duration: 5000,
      frameRate: {
        expected: 60,
        min: 30,
        max: 60
      }
    });
  }
}
```

### Rationale

代码中将动画帧率锁定在 144 FPS，对设备资源消耗过大，可能会影响设备其他应用的性能表现。

## Example 12: `pair_aa3c320fd30ec87c`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
const rateConfig = {
  expected: 150,
  min: 150,
  max: 150
};
displaySync.setRateRange(rateConfig);
```

### Repair pattern

```arkts
const rateConfig = {
  expected: 60,
  min: 30,
  max: 60
};
displaySync.setRateRange(rateConfig);
```

### Rationale

此配置将帧率锁定在 150 FPS，过高的帧率会导致设备性能下降，尤其是在较弱的硬件上。

## Example 13: `pair_c625d59193d8ee8f`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
const frameController = new FrameRateController();
frameController.setRange(240, 240, 240);
```

### Repair pattern

```arkts
const frameController = new FrameRateController();
frameController.setRange(60, 30, 60);
```

### Rationale

为帧率设置过高的范围，锁定在 240 FPS，将对 CPU 和 GPU 导致巨大的性能负担。

## Example 14: `pair_cc81fd21ba921bc0`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
function processFrame() {
  videoConfig = {
    fps: {
      min: 210,
      expected: 210,
      max: 210
    }
  };
  videoPlayer.configure(videoConfig);
}
```

### Repair pattern

```arkts
function processFrame() {
  videoConfig = {
    fps: {
      min: 30,
      expected: 60,
      max: 60
    }
  };
  videoPlayer.configure(videoConfig);
}
```

### Rationale

锁定帧率为 210 FPS，可能导致设备在运行其他应用时速度减慢。

## Example 15: `pair_f5a0cc18b744198f`

不允许锁定最高帧率运行。

### Triggering pattern

```arkts
let videoSettings = {
  expectedFrameRate: 200,
  minFrameRate: 200,
  maxFrameRate: 200
};
mediaPlayer.setRateOptions(videoSettings);
```

### Repair pattern

```arkts
let videoSettings = {
  expectedFrameRate: 60,
  minFrameRate: 30,
  maxFrameRate: 60
};
mediaPlayer.setRateOptions(videoSettings);
```

### Rationale

锁定帧率为 200 FPS 对于大多数设备来说负担过重，可能会导致性能降低。
