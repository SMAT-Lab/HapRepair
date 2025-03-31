@Entry
@Component
struct App {
  build() {
      Stack() {
        Column() {
          Column({ space: 18 }) {
            if (this.recordingStatus === 0) {
              ForEach(this.sidebarList_1, (sidebar: Resource, index: number) => {
                Image($r('app.media.app_icon'))
                  .width(28)
                  .height(28)
                  .objectFit(ImageFit.Contain)
                if (index === 1) {
                  Divider()
                    .vertical(false)
                    .height(1)
                    .width(22)
                    .color($r('app.color.COLOR_FFFFFF'))
                    .margin({ right: 4 })
                }
              }, (sidebar: Resource, index: number) => index)
            } else if (this.recordingStatus === 2) {
              ForEach(this.sidebarList_2, (sidebar: Resource, index: number) => {
                Image($r('app.media.app_icon'))
                  .width(28)
                  .height(28)
                  .objectFit(ImageFit.Contain)
                if (index === 0) {
                  Divider()
                    .vertical(false)
                    .height(1)
                    .width(22)
                    .color($r('app.color.COLOR_FFFFFF'))
                    .margin({ right: 4 })
                }
                if (index === 1) {
                  Text($r('app.string.Wen'))
                    .textAlign(TextAlign.Center)
                    .fontColor($r('app.color.COLOR_FFFFFF'))
                    .fontSize(22)
                    .fontFamily($r('app.string.Font_family_medium'))
                    .margin({ right: 4 })
                }
              }, (sidebar: Resource, index: number) => index)
            }
          }
          .width('100%')
          .height('50%')
        }
        .width('100%')
        .height('100%')
      }
      .width('100%')
      .height('91%')
      .borderRadius(12)

  }
}