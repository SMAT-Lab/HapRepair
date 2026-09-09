@Entry
@Component
struct FadingNotice {
  @State visible: boolean = true;
  @State noticeOpacity: number = 1;

  build() {
    Column() {
      if (this.visible) {
        Text('Saved')
          .opacity(this.noticeOpacity)
          .transition(TransitionEffect.OPACITY.animation({ duration: 500 }))
      }
      Button('Toggle')
        .onClick(() => {
          this.visible = !this.visible;
        })
    }
  }
}
