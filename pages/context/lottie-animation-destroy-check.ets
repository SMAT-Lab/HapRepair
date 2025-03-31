
    Canvas(this.politeChickyController)
      .width(160)
      .height(160)
      .backgroundColor(Color.Gray)
      .borderRadius(3)
      .onReady(() => {
        //告警
        this.animateItem = lottie.loadAnimation({
          container: this.politeChickyController,
          renderer: 'canvas',
          loop: true,
          autoplay: true,
          name: this.politeChicky,
          path: this.politeChickyPath,
          initialSegment: [FRAME_START, FRAME_END]
        })
      })

