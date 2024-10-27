import { hiTraceMeter } from '@kit.PerformanceAnalysisKit'

@Entry
@Component
struct GestureHandlerComponent {
  @State xPosition: number = 0;
  @State yPosition: number = 0;

  // 初始化手势的设置
  private gestureOptions: PanGestureOptions = new PanGestureOptions({
    direction: PanDirection.All, // 允许所有方向的拖动
    speed: 1.0,
  });

  // 开始手势追踪的逻辑
  private startTracking(event: GestureEvent): void {
    console.info('Gesture started');
    hiTraceMeter.startTrace("GestureTrace", 1);
  }

  // 拖动手势更新时的逻辑
  private updatePosition(event: GestureEvent): void {
    if (event) {
      this.xPosition += event.offsetX;
      this.yPosition += event.offsetY;
    }
  }

  // 手势结束后的逻辑
  private endTracking(): void {
    console.info('Gesture ended');
    hiTraceMeter.finishTrace("GestureTrace", 1);
  }

  // 手势设置修改器，用于调整手势的触发条件
  private adjustGestureTrigger(): void {
    this.gestureOptions.setDistance(300);
    console.info('Gesture distance threshold updated to 300');
  }

  build() {
    Column() {
      Column() {
        Text('拖动手势:\nX: ' + this.xPosition + '\nY: ' + this.yPosition)
      }
      .width(250)
      .height(250)
      .border({ width: 2, color: Color.Blue })
      .padding(20)
      .translate({ x: this.xPosition, y: this.yPosition })
      .gesture(
        PanGesture(this.gestureOptions)
          .onActionStart(this.startTracking)
          .onActionUpdate(this.updatePosition)
          .onActionEnd(this.endTracking)
      )

      Button('调整手势触发条件')
        .onClick(() => {
          this.adjustGestureTrigger();
        })
    }
    .padding(20)
    .margin({ top: 30 })
    .alignItems(HorizontalAlign.Center)
  }
}
