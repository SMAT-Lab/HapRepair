@Component
struct SwiperComponent {
  handleSwiperChange(index: number) {
    // 处理轮播图切换的逻辑
  }

  build() {
    Swiper()
      .onChange((index) => {
        this.handleSwiperChange(index)
      })
  }
}
