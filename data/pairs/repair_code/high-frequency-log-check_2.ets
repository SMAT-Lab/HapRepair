@Entry
@Component
struct Index {
  build() {
    Row(){
      Image("XXX")
        .width(200)
        .height(100)
        .draggable(true)
        .margin({ left: 15 })
        .border({ color: Color.Black, width: 1 })
        .onDragMove(() => {
          const TAG = 'onDragMove';
        })
    }
  }
}