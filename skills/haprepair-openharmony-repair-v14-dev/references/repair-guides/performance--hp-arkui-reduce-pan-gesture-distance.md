# @performance/hp-arkui-reduce-pan-gesture-distance

Static repair references: 2. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_c87e18bdc2f78e78`

建议设置合理的拖动距离

### Triggering pattern

```arkts
import { hiTraceMeter } from '@kit.PerformanceAnalysisKit'

@Entry
@Component
struct PanGestureExample {
  @State offsetX: number = 0
  @State offsetY: number = 0
  @State positionX: number = 0
  @State positionY: number = 0
  private panOption: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.Left | PanDirection.Right })

  build() {
    Column() {
      Column() {
        Text('PanGesture offset:\nX: ' + this.offsetX + '\n' + 'Y: ' + this.offsetY)
      }
      .height(200)
      .width(300)
      .padding(20)
      .border({ width: 3 })
      .margin(50)
      .translate({ x: this.offsetX, y: this.offsetY, z: 0 })
      // 左右拖动触发该手势事件
      .gesture(
        PanGesture(this.panOption)
          .onActionStart((event: GestureEvent) => {
            console.info('Pan start')
            hiTraceMeter.startTrace("PanGesture", 1)
          })
          .onActionUpdate((event: GestureEvent) => {
            if (event) {
              this.offsetX = this.positionX + event.offsetX
              this.offsetY = this.positionY + event.offsetY
            }
          })
          .onActionEnd(() => {
            this.positionX = this.offsetX
            this.positionY = this.offsetY
            console.info('Pan end')
            hiTraceMeter.finishTrace("PanGesture", 1)
          })
      )

      Button('修改PanGesture触发条件')
        .onClick(() => {
          // 设定的距离超过阈值10
          this.panOption.setDistance(100)
        })
    }
  }
}
```

### Repair pattern

```arkts
import { hiTraceMeter } from '@kit.PerformanceAnalysisKit'

@Entry
@Component
struct PanGestureExample {
  @State offsetX: number = 0
  @State offsetY: number = 0
  @State positionX: number = 0
  @State positionY: number = 0
  private panOption: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.Left | PanDirection.Right })

  build() {
    Column() {
      Column() {
        Text('PanGesture offset:\nX: ' + this.offsetX + '\n' + 'Y: ' + this.offsetY)
      }
      .height(200)
      .width(300)
      .padding(20)
      .border({ width: 3 })
      .margin(50)
      .translate({ x: this.offsetX, y: this.offsetY, z: 0 }) // 以组件左上角为坐标原点进行移动
      // 左右拖动触发该手势事件
      .gesture(
        PanGesture(this.panOption)
          .onActionStart((event: GestureEvent) => {
            console.info('Pan start')
            hiTraceMeter.startTrace("PanGesture", 1)
          })
          .onActionUpdate((event: GestureEvent) => {
            if (event) {
              this.offsetX = this.positionX + event.offsetX
              this.offsetY = this.positionY + event.offsetY
            }
          })
          .onActionEnd(() => {
            this.positionX = this.offsetX
            this.positionY = this.offsetY
            console.info('Pan end')
            hiTraceMeter.finishTrace("PanGesture", 1)
          })
      )

      Button('修改PanGesture触发条件')
        .onClick(() => {
          // 设定的距离在阈值10以内
          this.panOption.setDistance(4)
        })
    }
  }
}
```

### Rationale

pan gesture是手势拖动组件，设定的距离超过阈值10会造成操作极其不便的问题，需要更合理地设置拖动距离

## Example 2: `pair_e4aa37e18fd81de1`

建议设置合理的拖动距离

### Triggering pattern

```arkts
@Entry
@Component
struct DraggableBox {
  @State offsetX: number = 0;
  @State offsetY: number = 0;
  private panOption: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.All });

  build() {
    Column() {
      Rectangle()
        .width(100)
        .height(100)
        .backgroundColor(Color.Blue)
        .translate({ x: this.offsetX, y: this.offsetY })
        .gesture(
          PanGesture(this.panOption)
            .onActionStart(() => {
              console.info('Pan started');
            })
            .onActionUpdate((event: GestureEvent) => {
              if (event) {
                this.offsetX += event.offsetX;
                this.offsetY += event.offsetY;
              }
            })
            .onActionEnd(() => {
              console.info('Pan ended');
            })
        )

      Button('设置拖动距离')
        .onClick(() => {
          // 设定的距离超过阈值10
          this.panOption.setDistance(50);
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct DraggableBox {
  @State offsetX: number = 0;
  @State offsetY: number = 0;
  private panOption: PanGestureOptions = new PanGestureOptions({ direction: PanDirection.All });

  build() {
    Column() {
      Rectangle()
        .width(100)
        .height(100)
        .backgroundColor(Color.Blue)
        .translate({ x: this.offsetX, y: this.offsetY })
        .gesture(
          PanGesture(this.panOption)
            .onActionStart(() => {
              console.info('Pan started');
            })
            .onActionUpdate((event: GestureEvent) => {
              if (event) {
                this.offsetX += event.offsetX;
                this.offsetY += event.offsetY;
              }
            })
            .onActionEnd(() => {
              console.info('Pan ended');
            })
        )

      Button('设置拖动距离')
        .onClick(() => {
          // 设定的距离在阈值10以内
          this.panOption.setDistance(5);
        })
    }
  }
}
```

### Rationale

在上述代码中，setDistance(50) 将拖动手势的触发距离设置为 50，超过了推荐的阈值 10，这会导致用户需要拖动较长距离才能触发手势，影响操作体验。应将拖动距离设置在合理范围内，例如 5，以提高响应速度和用户体验。
