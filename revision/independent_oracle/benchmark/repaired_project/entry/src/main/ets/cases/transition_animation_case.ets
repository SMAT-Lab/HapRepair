@Entry
@Component
struct FadingNotice {
  @State visible: boolean = true;

  build() {
    Column() {
      if (this.visible) {
        Text('Saved')
          .id('saved-notice')
          .transition(TransitionEffect.OPACITY.animation({ duration: 500 }))
      }
      Button('Toggle')
        .onClick(() => {
          this.visible = !this.visible;
        })
    }
  }
}
