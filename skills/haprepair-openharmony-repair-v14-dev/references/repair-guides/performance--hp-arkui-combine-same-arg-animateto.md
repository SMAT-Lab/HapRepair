# @performance/hp-arkui-combine-same-arg-animateto

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_009aa747e93bbdd7`

建议动画参数相同时使用同一个animateTo

### Triggering pattern

```arkts
@Entry
@Component
struct SizeChanger {
  @State width: number = 100;
  @State height: number = 100;

  increaseWidth() {
    animateTo({ duration: 800, curve: Curve.EaseInOut }, () => {
      this.width += 50;
    });
  }

  increaseHeight() {
    animateTo({ duration: 800, curve: Curve.EaseInOut }, () => {
      this.height += 50;
    });
  }

  build() {
    Column() {
      Rectangle()
        .width(this.width)
        .height(this.height)
        .backgroundColor(Color.Blue)
      Button('Increase Size')
        .onClick(() => {
          this.increaseWidth();
          this.increaseHeight();
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct SizeChanger {
  @State width: number = 100;
  @State height: number = 100;

  increaseSize() {
    animateTo({ duration: 800, curve: Curve.EaseInOut }, () => {
      this.width += 50;
      this.height += 50;
    });
  }

  build() {
    Column() {
      Rectangle()
        .width(this.width)
        .height(this.height)
        .backgroundColor(Color.Blue)
      Button('Increase Size')
        .onClick(() => {
          this.increaseSize();
        })
    }
  }
}
```

### Rationale

将 increaseWidth() 和 increaseHeight() 中的动画合并到一个 increaseSize() 方法中，避免重复使用相同的动画参数，多次调用 animateTo。这样可以使尺寸的变化同步进行，提高动画的流畅性和性能。

## Example 2: `pair_0ea4de657a30e263`

建议动画参数相同时使用同一个animateTo

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State textWidth: number = 200;
  @State color: Color = Color.Red;
  
  func1() {
    animateTo({ curve: Curve.Sharp, duration: 1000 }, () => {
      this.textWidth = (this.textWidth === 100 ? 200 : 100);
    });
  }
  
  func2() {
    animateTo({ curve: Curve.Sharp, duration: 1000 }, () => {
      this.color = (this.color === Color.Yellow ? Color.Red : Color.Yellow);
    });
  }
  
  build() {
    Column() {
      Row()
        .width(this.textWidth)
        .height(10)
        .backgroundColor(this.color)
      Text('click')
        .onClick(() => {
          this.func1();
          this.func2();
        })
    }
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MyComponent {
  @State textWidth: number = 200;
  @State color: Color = Color.Red;
  
  func() {
    animateTo({ curve: Curve.Sharp, duration: 1000 }, () => {
      this.textWidth = (this.textWidth === 100 ? 200 : 100);
      this.color = (this.color === Color.Yellow ? Color.Red : Color.Yellow);
    });
  }
  
  build() {
    Column() {
      Row()
        .width(this.textWidth)
        .height(10)
        .backgroundColor(this.color)
      Text('click')
        .onClick(() => {
          this.func();
        })
    }
    .width('100%')
    .height('100%')
  }
}
```

### Rationale

func1和func2中的animateTo的动画参数相同，合并成同一个动画

## Example 3: `pair_18920b26d2d2a63a`

建议动画参数相同时使用同一个animateTo

### Triggering pattern

```arkts
@Entry
@Component
struct MoveComponent {
  @State posX: number = 0;
  @State posY: number = 0;

  moveRight() {
    animateTo({ duration: 500, curve: Curve.Linear }, () => {
      this.posX += 50;
    });
  }

  moveDown() {
    animateTo({ duration: 500, curve: Curve.Linear }, () => {
      this.posY += 50;
    });
  }

  build() {
    Column() {
      Image('icon.png')
        .position({ x: this.posX, y: this.posY })
      Button('Move')
        .onClick(() => {
          this.moveRight();
          this.moveDown();
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MoveComponent {
  @State posX: number = 0;
  @State posY: number = 0;

  move() {
    animateTo({ duration: 500, curve: Curve.Linear }, () => {
      this.posX += 50;
      this.posY += 50;
    });
  }

  build() {
    Column() {
      Image('icon.png')
        .position({ x: this.posX, y: this.posY })
      Button('Move')
        .onClick(() => {
          this.move();
        })
    }
  }
}
```

### Rationale

原代码中 moveRight() 和 moveDown() 方法中的 animateTo 动画参数相同，可以合并为一个 move() 方法，将两个状态变量的更新放在同一个 animateTo 回调中，以提高性能并保证动画同步。

## Example 4: `pair_21c9bb743e141d61`

建议动画参数相同时使用同一个animateTo

### Triggering pattern

```arkts
@Entry
@Component
struct RotateComponent {
  @State angle1: number = 0;
  @State angle2: number = 0;

  rotateFirst() {
    animateTo({ duration: 700, curve: Curve.Sharp }, () => {
      this.angle1 += 45;
    });
  }

  rotateSecond() {
    animateTo({ duration: 700, curve: Curve.Sharp }, () => {
      this.angle2 += 45;
    });
  }

  build() {
    Column() {
      Image('image1.png')
        .rotate(this.angle1)
      Image('image2.png')
        .rotate(this.angle2)
      Button('Rotate Images')
        .onClick(() => {
          this.rotateFirst();
          this.rotateSecond();
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct RotateComponent {
  @State angle1: number = 0;
  @State angle2: number = 0;

  rotateBoth() {
    animateTo({ duration: 700, curve: Curve.Sharp }, () => {
      this.angle1 += 45;
      this.angle2 += 45;
    });
  }

  build() {
    Column() {
      Image('image1.png')
        .rotate(this.angle1)
      Image('image2.png')
        .rotate(this.angle2)
      Button('Rotate Images')
        .onClick(() => {
          this.rotateBoth();
        })
    }
  }
}
```

### Rationale

将两个旋转动画合并到 rotateBoth() 方法中，使用一个 animateTo 调用，同时更新 angle1 和 angle2。这样可以使两个图片的旋转动作同步，避免动画参数重复定义，提升性能。

## Example 5: `pair_a33998727a4eef28`

建议动画参数相同时使用同一个animateTo

### Triggering pattern

```arkts
@Entry
@Component
struct OpacityChanger {
  @State opacity1: number = 1;
  @State opacity2: number = 1;

  fadeOutFirst() {
    animateTo({ duration: 600 }, () => {
      this.opacity1 = 0;
    });
  }

  fadeOutSecond() {
    animateTo({ duration: 600 }, () => {
      this.opacity2 = 0;
    });
  }

  build() {
    Column() {
      Text('First Text')
      Text('Second Text')
      Button('Fade Out')
        .onClick(() => {
          this.fadeOutFirst();
          this.fadeOutSecond();
        })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct OpacityChanger {
  @State opacity1: number = 1;
  @State opacity2: number = 1;

  fadeOutBoth() {
    animateTo({ duration: 600 }, () => {
      this.opacity1 = 0;
      this.opacity2 = 0;
    });
  }

  build() {
    Column() {
      Text('First Text')
      Text('Second Text')
      Button('Fade Out')
        .onClick(() => {
          this.fadeOutBoth();
        })
    }
  }
}
```

### Rationale

将两个文本的淡出动画合并到 fadeOutBoth() 方法中，使用一个 animateTo 调用，同时更新 opacity1 和 opacity2。这样可以确保两个文本同步淡出，提升用户体验并优化性能。
