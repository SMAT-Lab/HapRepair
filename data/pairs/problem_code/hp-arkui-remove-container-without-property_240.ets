@Entry
@Component
struct ProfileCard {
  build() {
    Column() {
      Stack() {
        Stack() {
          Image($r('app.media.avatar'))
            .width(100)
            .height(100)
        }
        .backgroundColor(Color.White)
      }
      .padding(10)
    }
    .borderRadius(8)
    .backgroundColor(Color.Gray)
  }
}
