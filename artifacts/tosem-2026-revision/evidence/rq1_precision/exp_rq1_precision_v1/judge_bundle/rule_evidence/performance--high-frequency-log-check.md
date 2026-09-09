# @performance/high-frequency-log-check

Static repair references: 20. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_0e5a48e69b552ae9`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Slider()
      .onActionUpdate(() => {
        hilog.warn(1001, 'App', 'onActionUpdate')
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Slider()
      .onActionUpdate(() => {
        const TAG = 'onActionUpdate';
    })
  }
}
```

### Rationale

onActionUpdate 是被频繁调用的函数。在滑块更新事件中，每次更新都会触发这个函数。添加日志会导致性能下降。

## Example 2: `pair_2252a5961f5e7e48`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Row(){
      Image("XXX")
        .width(200)
        .height(100)
        .draggable(true)
        .margin({ left: 15 })
        .border({ color: Color.Black, width: 1 })
        .onDragMove(() => {
          hilog.info(1001, 'Index', 'onDragMove')
      })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct Index {
  build() {
    Row(){
      Image("XXX")
        .width(200)
        .height(100)
        .draggable(true)
        .margin({ left: 15 })
        .border({ color: Color.Black, width: 1 })
        .onDragMove(() => {
          const TAG = 'onDragMove';
        })
    }
  }
}
```

### Rationale

在拖拽组件中，onDragMove 函数会在每次拖动时被触发。如果在 onDragMove 函数中使用 hilog.warn 进行日志记录，就会导致每次拖动操作都进行一次日志记录，这样的高频操作会对性能造成负面影响。通过移除高频函数中的 hilog 调用，减少了不必要的日志输出，从而降低了对系统性能的影响。

## Example 3: `pair_25fa1e256ea35bd0`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct ControlPanel {
  build() {
    DatePicker()
      .onDateChange(() => {
        hilog.warn(8004, 'ControlPanel', 'onValueChange')
      })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ControlPanel {
  build() {
    DatePicker()
      .onDateChange(() => {
        const ACTION = 'onValueChange';
      })
  }
}
```

### Rationale

onDateChange 是高频事件，切换值时多次调用。加入日志会对性能造成负面影响。

## Example 4: `pair_2c3295e95609ca22`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Home {
  build() {
    TextInput()
      .onTextChange(() => {
        hilog.debug(5001, 'Home', 'onTextChange')
      })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct Home {
  build() {
    TextInput()
      .onTextChange(() => {
        const TAG = 'onTextChange';
      })
  }
}
```

### Rationale

onTextChange 是一个高频触发事件，每次输入更新都会触发此函数。使用 hilog.debug 记录事件可能会让性能受损。

## Example 5: `pair_2e795458d99fed5d`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    ListItem()
      .onMouse(() => {
        hilog.info(2001, 'App', 'onMouse')
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    ListItem()
      .onMouse(() => {
        const TAG = 'onMouse';
    })
  }
}
```

### Rationale

onMouse 是一个高频触发的事件。使用 hilog.info 记录每一次的鼠标事件操作会影响性能。

## Example 6: `pair_323fac8dda8b80c9`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    List() {
      Text('Item')
    }.onItemDragMove(() => {
      hilog.debug(1001, 'App', 'onItemDragMove')
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List() {
      Text('Item')
    }.onItemDragMove(() => {
      const TAG = 'onItemDragMove';
    })
    .width("100%")
    .height("100%")
  }
}
```

### Rationale

在项目中，onItemDragMove 是一个高频事件。在这个情况下，每次拖动项目都会触发 onItemDragMove 函数。如果在该函数中使用 hilog.debug 进行日志记录，会导致大量日志输出，进而影响应用性能。

## Example 7: `pair_3a82537afa39609e`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
// Test.ets
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
    build() {
            Column() {
                Scroll()
                    .onScroll(() => {
                        hilog.info(1001, 'Index', 'onScroll') // Avoid printing logs
                })
            }
    }
}
```

### Repair pattern

```arkts
// Test.ets
@Entry
@Component
struct Index {
  build() {
      Column() {
        Scroll()
          .onScroll(() => {
            const TAG = 'onScroll';
          })
      }
  }
}
```

### Rationale

在滚动组件中触发滚动属于高频事件，而 onScroll 函数会在每次滚动时被触发。如果在 onScroll 函数中使用 hilog.info 进行日志记录，就会导致每次滚动操作都进行一次日志记录，这样的高频操作会对性能造成负面影响。通过移除高频函数中的 hilog 调用，减少了不必要的日志输出，从而降低了对系统性能的影响。

## Example 8: `pair_3fbbdf46efac42c3`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Toggle()
      .onActionUpdate(() => {
        hilog.debug(3001, 'App', 'Toggle Action Update')
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Toggle()
      .onActionUpdate(() => {
        const state = 'Toggle Action Update';
    })
  }
}
```

### Rationale

当 Toggle 的状态发生变化时，onActionUpdate 会被持续调用。使用 hilog.debug 会导致过多的日志输出，从而降低性能。

## Example 9: `pair_464d9a7bb4986d54`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Button("")
      .width(200)
      .height(100)
      .onMouse(() => {
        hilog.info(1001, 'Index', 'onMouse')
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
    Button("")
      .width(200)
      .height(100)
      .onMouse(() => {
        const TAG = 'onMouse';
      })
  }
}
```

### Rationale

在鼠标事件中，onMouse 函数会在每次鼠标操作时被触发。如果在 onMouse 函数中使用 hilog.error 进行日志记录，就会导致每次鼠标操作都进行一次日志记录，这样的高频操作会对性能造成负面影响。通过移除高频函数中的 hilog 调用，减少了不必要的日志输出，从而降低了对系统性能的影响。

## Example 10: `pair_4b8663f2547fceb9`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Dashboard {
  build() {
    Scroll()
      .onScrollStart(() => {
        hilog.info(6002, 'Dashboard', 'onScroll')
      })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct Dashboard {
  build() {
    Scroll()
      .onScrollStart(() => {
        const LABEL = 'onScroll';
      })
  }
}
```

### Rationale

onItemMove 是高频事件，在用户滚动时频繁调用。使用 hilog.info 会严重影响应用性能。

## Example 11: `pair_5ac0621de1f4db10`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Row() {
      Slider({ value: 0, min: 0, max: 100 })
        .enabled(false)
        .height(4)
        .width(100)
        .trackThickness(3)
        .blockColor(Color.Red)
        .blockSize({ width: 4, height: 4 })
        .onVisibleAreaChange([0.0, 1.0], (isVisible: boolean, currentRatio: number) => {
          hilog.info(1001, 'Index', 'onVisibleAreaChange') // Avoid printing logs
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct Index {
  build() {
    Row() {
      Slider({ value: 0, min: 0, max: 100 })
        .enabled(false)
        .height(4)
        .width(100)
        .trackThickness(3)
        .blockColor(Color.Red)
        .blockSize({ width: 4, height: 4 })
        .onVisibleAreaChange([0.0, 1.0], (isVisible: boolean, currentRatio: number) => {
          const TAG = 'onVisibleAreaChange';
        })
    }
  }
}
```

### Rationale

在可见区域变化事件中，onVisibleAreaChange 函数会在每次可见区域变化时被触发。如果在 onVisibleAreaChange 函数中使用 hilog.info 进行日志记录，就会导致每次可见区域变化操作都进行一次日志记录，这样的高频操作会对性能造成负面影响。通过移除高频函数中的 hilog 调用，减少了不必要的日志输出，从而降低了对系统性能的影响。

## Example 12: `pair_878934088d966172`

不建议在高频函数中使用Hilog或者其它打印日志的方式。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Canvas(()=>{})
      .width('100%')
      .height('100%')
      .onAreaChange(() => {
        hilog.error(1001, 'App', 'onAreaChange')
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Canvas(()=>{})
      .width('100%')
      .height('100%')
      .onAreaChange(() => {
        const TAG = 'onAreaChange';
    })
  }
}
```

### Rationale

onAreaChange 是高频事件之一。在布局中，每次区域变化都会触发这个函数。如果在该函数中使用 hilog.error 进行日志记录，可能导致性能问题。

## Example 13: `pair_9dcb3d82d58ff74e`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Scroll(this.scroller) {
      Column() {
        Text(this.message)
        .fontSize(50)
        .fontWeight(FontWeight.Bold)
      }.onScroll((xOffset: number, yOffset: number) => {
        console.info(xOffset + ' ' + yOffset)
      })
    }
  }
}
```

### Repair pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Scroll(this.scroller) {
      Column() {
        Text(this.message)
        .fontSize(50)
        .fontWeight(FontWeight.Bold)
      }.onScroll((xOffset: number, yOffset: number) => {
        const TAG = xOffset + ' ' + yOffset
      })
    }
  }
}
```

### Rationale

在滚动组件中触发滚动属于高频事件，而 onScroll 函数会在每次滚动时被触发。如果在 onScroll 函数中使用console.log来进行记录会产生大量的日志，显著地影响了app的性能

## Example 14: `pair_a1c5dd86c16c651c`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      Text('1')
        .fontWeight(FontWeight.Bold)
        .backgroundColor('black')
        .fontColor('white')
        .fontSize(200)
        .width(200)
        .height(200)
        .lineHeight(200)
        .textAlign(TextAlign.Center)
    }
    .onTouch((event: TouchEvent) => {
      console.log(JSON.stringify(event.touches))
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  Row() {
    Text('1')
      .fontWeight(FontWeight.Bold)
      .backgroundColor('black')
      .fontColor('white')
      .fontSize(200)
      .width(200)
      .height(200)
      .lineHeight(200)
      .textAlign(TextAlign.Center)

      .onTouch((event: TouchEvent) => {
        const text = JSON.stringify(event.touches)
      })
  }
}
```

### Rationale

在滚动组件中触发滚动属于高频事件，而 onScroll 函数会在每次滚动时被触发。如果在 onScroll 函数中使用console.log来进行记录会产生大量的日志，显著地影响了app的性能

## Example 15: `pair_b4758c6cd58c6057`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct UserInterface {
  build() {
    List()
      .onItemMove(() => {
        hilog.error(7003, 'UserInterface', 'onItemMove')
      })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct UserInterface {
  build() {
    List()
      .onItemMove(() => {
        const EVENT = 'onItemMove';
      })
      .width("100%")
      .height("100%")
  }
}
```

### Rationale

onItemMove 是一个高频事件，用户选择时接连触发。使用 hilog.error 会导致性能瓶颈。

## Example 16: `pair_b766bd6f34d31667`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Stack() {
      Column() {
        LoadingPanel()
      }
      .width('100%')
      .height('100%')

      Row() {
        if (this.thumbnail !== null && this.thumbnail !== undefined) {
          Image(this.thumbnail)
            .rotate({
              x: 0,
              y: 0,
              z: 1,
              angle: 0
            })
            .onComplete((): void => {
              Log.info(TAG,
                'onComplete finish, index: ' + this.item.index + ', item: ' + JSON.stringify(this.item) + ', uri: ' +
                this.thumbnail + '.');
            })
            .onError((): void => {
              Log.error(TAG, 'image show error ' + this.thumbnail + ' ' + this.item.width + ' ' + this.item.height);
            })
        }

      }
      .width('100%')
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Stack() {
      Column() {
        LoadingPanel()
      }
      .width('100%')
      .height('100%')

      Row() {
        if (this.thumbnail !== null && this.thumbnail !== undefined) {
          Image(this.thumbnail)
            .rotate({
              x: 0,
              y: 0,
              z: 1,
              angle: 0
            })
            .onComplete((): void => {
              let info = TAG +
                'onComplete finish, index: ' + this.item.index + ', item: ' + JSON.stringify(this.item) + ', uri: ' +
              this.thumbnail + '.';
            })
            .onError((): void => {
              let error = TAG + 'image show error ' + this.thumbnail + ' ' + this.item.width + ' ' + this.item.height;
            })
        }

      }
      .width('100%')
      .height('100%')
    }
  }
}
```

### Rationale

在滚动组件中触发滚动属于高频事件，而 onScroll 函数会在每次滚动时被触发。如果在 onScroll 函数中使用Log来进行记录会产生大量的日志，显著地影响了app的性能

## Example 17: `pair_d46a40c932bf009f`

不建议在高频函数中使用Hilog或者其它打印日志的方式。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Canvas(()=>{})
      .width('100%')
      .height('100%')
      .onScroll((xOffset: number, yOffset: number) => {
        console.info(1001, 'App', 'onScroll')
    })
  }
}
```

### Repair pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    Canvas(()=>{})
      .width('100%')
      .height('100%')
      .onScroll((xOffset: number, yOffset: number) => {
        const TAG = 'onScroll'
    })
  }
}
```

### Rationale

onScroll 是高频事件之一。在布局中，每次滚动都会触发这个函数。如果在该函数中使用 console.info 进行日志记录，可能导致性能问题。

## Example 18: `pair_dcb78cb20212adfa`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct MainScreen {
  build() {
    Swipe()
      .onChange(() => {
        hilog.debug(9005, 'MainScreen', 'onChange')
      })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MainScreen {
  build() {
    Swipe()
      .onChange(() => {
        const STATUS = 'onChange';
      })
  }
}
```

### Rationale

onChange 是高频触发事件。在滑动更新中，记录日志可能会导致大量日志产生，性能下降。

## Example 19: `pair_dee8b38e16d2f258`

不建议在高频函数中使用Hilog。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct App {
  build() {
    List() {
      Text('Item')
    }.onItemDragMove(() => {
      hilog.debug(1001, 'App', 'onItemDragMove')
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List() {
      Text('Item')
    }.onItemDragMove(() => {
      const TAG = 'onItemDragMove';
    })
  }
}
```

### Rationale

在项目中，onItemDragMove 是一个高频事件。在这个情况下，每次拖动项目都会触发 onItemDragMove 函数。如果在该函数中使用 hilog.debug 进行日志记录，会导致大量日志输出，进而影响应用性能。

## Example 20: `pair_ffb5205e70660434`

不建议在高频函数中使用Hilog。

高频函数包括：onTouch、onItemDragMove、onDragMove、onMouse、onVisibleAreaChange、onAreaChange、onScroll、onActionUpdate。

### Triggering pattern

```arkts
import hilog from '@ohos.hilog';
@Entry
@Component
struct Index {
  build() {
    Button("XXX")
      .width(200)
      .height(100)
      .onTouch(() => {
        hilog.info(1001, 'Index', 'onTouch') // Avoid printing logs
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
    Button("XXX")
      .width(200)
      .height(100)
      .onTouch(() => {
        const TAG = 'onTouch';
      })
  }
}
```

### Rationale

在触摸事件中，onTouch 函数会在每次触摸时被触发。如果在 onTouch 函数中使用 hilog.debug 进行日志记录，就会导致每次触摸操作都进行一次日志记录，这样的高频操作会对性能造成负面影响。通过移除高频函数中的 hilog 调用，减少了不必要的日志输出，从而降低了对系统性能的影响。
