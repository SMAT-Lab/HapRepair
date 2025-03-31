
  build() {
    // Swiper组件没有使用OnAnimationStart进行预加载
    Swiper(this.swiperController) {
      LazyForEach(this.data, (item: MyObject, index?: number) => {
        // 源码文件，请以工程实际为准
        PhotoItem({ myIndex: index, dataSource: this.data })
      }, (item: MyObject) => item.id)
    }