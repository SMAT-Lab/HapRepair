# @performance/hp-arkui-image-async-load

Static repair references: 3. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_2129cd15646ee7a8`

建议大图片使用异步加载

### Triggering pattern

```arkts
@Entry
@Component
struct BackgroundImageComponent {
  build() {
    Stack() {
      // 网络大图片
      Image('https://example.com/large-background.jpg')
        .width('100%')
        .height('100%')
        .syncLoad(true) // 同步加载大图片
      Text('欢迎使用我们的应用')
        .fontSize(24)
        .fontColor(Color.White)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct BackgroundImageComponent {
  build() {
    Stack() {
      // 网络大图片
      Image('https://example.com/large-background.jpg')
        .width('100%')
        .height('100%')
        // 移除 syncLoad，使用异步加载
      Text('欢迎使用我们的应用')
        .fontSize(24)
        .fontColor(Color.White)
    }
  }
}
```

### Rationale

对于需要加载的大尺寸背景图片，应该使用异步加载，避免使用 syncLoad(true)，以防止阻塞 UI 线程，保证界面其他元素的正常显示。

## Example 2: `pair_7d42e8d5a5175666`

建议大图片使用异步加载

### Triggering pattern

```arkts
@Entry
@Component
struct LargeImageComponent {
  build() {
    Column() {
      // 本地高清图片 hd_image.png
      Image($r('app.media.hd_image'))
        .width('100%')
        .height(200)
        .syncLoad(true) // 同步加载大图片
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct LargeImageComponent {
  build() {
    Column() {
      // 本地高清图片 hd_image.png
      Image($r('app.media.hd_image'))
        .width('100%')
        .height(200)
        // 移除 syncLoad，使用异步加载
    }
  }
}
```

### Rationale

对于大图片，建议使用异步加载方式，避免使用 syncLoad(true)。同步加载大图片可能导致主线程阻塞，影响应用性能和用户体验。

## Example 3: `pair_8c6a017ffbe2336a`

建议大图片使用异步加载

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  build() {
    Row() {
      // 本地图片4k.png
      Image($r('app.media.4k'))
        .border({ width: 1 })
        .borderStyle(BorderStyle.Dashed)
        .height(100)
        .width(100)
        .syncLoad(true)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MyComponent {
  build() {
    Row() {
      // 本地图片4k.png
      Image($r('app.media.4k'))
        .border({ width: 1 })
        .borderStyle(BorderStyle.Dashed)
        .height(100)
        .width(100)
    }
  }
}
```

### Rationale

大图片使用异步加载，不要使用syncLoad
