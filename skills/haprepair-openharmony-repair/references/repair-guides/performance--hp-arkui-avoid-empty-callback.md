# @performance/hp-arkui-avoid-empty-callback

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_29dccaf14820243a`

避免设置空的系统回调监听。

根据ArkUI编程规范，建议修改。

### Triggering pattern

```arkts
@Component
struct TextInputComponent {
  build() {
    TextInput()
      .onChange((value) => {
        // 无业务逻辑
      })
  }
}
```

### Repair pattern

```arkts
@Component
struct TextInputComponent {
  handleInputChange(value: string) {
    // 处理输入变化的逻辑
  }

  build() {
    TextInput()
      .onChange((value) => {
        this.handleInputChange(value)
      })
  }
}
```

### Rationale

避免设置空的系统回调监听。在上述代码中，onChange 回调函数未执行任何操作，应添加相应的业务逻辑函数 handleInputChange，以处理输入内容的变化。

## Example 2: `pair_59a4485a192f3baa`

避免设置空的系统回调监听。

根据ArkUI编程规范，建议修改。

### Triggering pattern

```arkts
@Component
struct MyComponent {
  build() {
    Button('Click', { type: ButtonType.Normal, stateEffect: true })
      .onClick(() => {
        // 无业务逻辑
      })
  }
}
```

### Repair pattern

```arkts
@Component
struct MyComponent {
  doSomething() {
    //业务逻辑
  }

  build() {
    Button('Click', { type: ButtonType.Normal, stateEffect: true })
      .onClick(() => {
        this.doSomething()
      })
  }
}
```

### Rationale

避免设置空的系统回调监听,这里onClick未执行任何逻辑，需要添加相应的执行函数

## Example 3: `pair_71387bd792925147`

避免设置空的系统回调监听。

根据ArkUI编程规范，建议修改。

### Triggering pattern

```arkts
@Component
struct ListComponent {
  build() {
    List()
      .onScroll(() => {
        // 无业务逻辑
      })
  }
}
```

### Repair pattern

```arkts
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
```

### Rationale

在 onScroll 回调中添加实际的处理函数 handleScrollEvent，以处理列表滚动事件，避免空的系统回调监听。

## Example 4: `pair_872d063b9fd25f6d`

避免设置空的系统回调监听。

根据ArkUI编程规范，建议修改。

### Triggering pattern

```arkts
@Component
struct GestureComponent {
  build() {
    Image('gesture.png')
      .gesture(TapGesture({ count: 2 })
        .onAction(() => {
          // 无业务逻辑
        })
      )
  }
}
```

### Repair pattern

```arkts
@Component
struct GestureComponent {
  handleDoubleTap() {
    // 处理双击手势的逻辑
  }

  build() {
    Image('gesture.png')
      .gesture(TapGesture({ count: 2 })
        .onAction(() => {
          this.handleDoubleTap()
        })
      )
  }
}
```

### Rationale

为双击手势的回调添加处理函数 handleDoubleTap，以处理用户的双击操作，避免设置空的系统回调监听。

## Example 5: `pair_b7f813690675770d`

避免设置空的系统回调监听。

根据ArkUI编程规范，建议修改。

### Triggering pattern

```arkts
@Component
struct SwiperComponent {
  build() {
    Swiper()
      .onChange((index) => {
        // 无业务逻辑
      })
  }
}
```

### Repair pattern

```arkts
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
```

### Rationale

在 onChange 回调中添加处理函数 handleSwiperChange，以处理轮播图索引变化的事件，避免空的系统回调监听。
