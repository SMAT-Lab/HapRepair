
      Image('resources/base/media/sss001.jpg').blur(1)
        .border({ width: 1 }).borderStyle(BorderStyle.Dashed).aspectRatio(1).width('90%').height(40)

      Text()
        .width('90%')
        .height(40)
        .fontSize(16)
        .backdropBlur(3)
        .backgroundImage('/pages/attrs/image/image.jpg')
        .backgroundImageSize({ width: 1200, height: 160 })
    }.width('100%').margin({ top: 5 })