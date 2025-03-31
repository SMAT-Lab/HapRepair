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