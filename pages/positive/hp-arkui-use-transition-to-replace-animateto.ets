@Entry
@Component
struct MyComponent {
  @State show: boolean = true;

  build() {
    Column() {
      Row() {
        if (this.show) {
          Text('value')
            // Set id to make transition interruptible
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