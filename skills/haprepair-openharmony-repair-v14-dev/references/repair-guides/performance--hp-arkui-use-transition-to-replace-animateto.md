# @performance/hp-arkui-use-transition-to-replace-animateto

Static repair references: 1. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_5210244a86e4f94b`

Use transition for component transition animation

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State mOpacity: number = 1;
  @State show: boolean = true;

  build() {
    Column() {
      Row() {
        if (this.show) {
          Text('value')
            .opacity(this.mOpacity)
        }
      }
      .width('100%')
      .height(100)
      .justifyContent(FlexAlign.Center)

      Text('toggle state')
        .onClick(() => {
          this.show = true;
          animateTo({
            duration: 1000, onFinish: () => {
              if (this.mOpacity === 0) {
                this.show = false;
              }
            }
          }, () => {
            this.mOpacity = this.mOpacity === 1 ? 0 : 1;
          })
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State show: boolean = true;

  build() {
    Column() {
      Row() {
        if (this.show) {
          Text('value')// Set id to make transition interruptible
            .id('myText')
            .transition(TransitionEffect.OPACITY.animation({ duration: 1000 }))
        }
      }.width('100%')
      .height(100)
      .justifyContent(FlexAlign.Center)

      Text('toggle state')
        .onClick(() => {
          // Through transition, animates the appearance or disappearance of transparency.
          this.show = !this.show;
        })
    }
  }
}
```

### Rationale

原来的代码中使用了animtaeTo函数，通过改变opacity来使得点击后Text('value')慢慢消失，这会造成帧率的丢失，建议使用.trainsition的属性来进行改变
