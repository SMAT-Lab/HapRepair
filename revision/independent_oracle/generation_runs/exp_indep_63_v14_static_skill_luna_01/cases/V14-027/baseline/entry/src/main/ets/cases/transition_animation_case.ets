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
      }
      Button('Toggle')
        .onClick(() => {
          this.visible = true;
          animateTo({
            duration: 500,
            onFinish: () => {
              if (this.noticeOpacity === 0) {
                this.visible = false;
              }
            }
          }, () => {
            this.noticeOpacity = this.noticeOpacity === 1 ? 0 : 1;
          });
        })
    }
  }
}
