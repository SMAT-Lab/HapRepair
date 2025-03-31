@Component
struct ListComponent {
  handleScrollEvent() {
    // 处理滚动事件的逻辑
  }

  build() {
    List()
      .onScroll(() => {
        this.handleScrollEvent()
      })
      .width("100%")
      .height("100%")
  }
}
