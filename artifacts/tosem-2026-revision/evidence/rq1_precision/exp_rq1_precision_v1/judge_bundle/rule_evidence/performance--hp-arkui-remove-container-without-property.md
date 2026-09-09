# @performance/hp-arkui-remove-container-without-property

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_1424c6c96ced77d5`

建议尽量减少视图嵌套层次

### Triggering pattern

```arkts
@Entry
@Component
struct ListExample {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Stack() {
            Stack() {
              Text('Item ' + item)
            }
          }
        }
      },(item:number)=>item.toString())
    }
    .cachedCount(2)
    .width("100%")
    .height('100%')
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ListExample {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Text('Item ' + item)
        }
      }, (item:number) => item.toString())
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
```

### Rationale

移除无属性的嵌套 Stack 容器，直接在 ListItem 中添加 Text 组件。

## Example 2: `pair_4bef7b293ceab35f`

建议尽量减少视图嵌套层次

### Triggering pattern

```arkts
@Entry
@Component
struct ProfileCard {
  build() {
    Column() {
      Stack() {
        Stack() {
          Image($r('app.media.avatar'))
            .width(100)
            .height(100)
        }
        .backgroundColor(Color.White)
      }
      .padding(10)
    }
    .borderRadius(8)
    .backgroundColor(Color.Gray)
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ProfileCard {
  build() {
    Column() {
      Image($r('app.media.avatar'))
        .width(100)
        .height(100)
        .backgroundColor(Color.White)
    }
    .padding(10)
    .borderRadius(8)
    .backgroundColor(Color.Gray)
  }
}
```

### Rationale

将 Stack 容器的 backgroundColor 属性上移到 Image 组件，移除冗余的嵌套 Stack 容器。

## Example 3: `pair_7c346ed67f43ad46`

建议尽量减少视图嵌套层次

### Triggering pattern

```arkts
@Entry
@Component
struct ImageGallery {
  build() {
    Scroll() {
      Flex() {
        Flex() {
          Flex() {
            Image($r('app.media.image1'))
            Image($r('app.media.image2'))
          }
        }
      }
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ImageGallery {
  build() {
    Scroll() {
      Flex() {
        Image($r('app.media.image1'))
        Image($r('app.media.image2'))
      }
    }
  }
}
```

### Rationale

移除多余的嵌套 Flex 容器，这些容器没有任何属性或样式，对布局没有影响。

## Example 4: `pair_c3de43660df422a8`

建议尽量减少视图嵌套层次

### Triggering pattern

```arkts
@Entry
@Component
struct ExampleComponent {
  build() {
    Column() {
      // 第一层无属性的 Column
      Column() {
        // 第二层无属性的 Column
        Column() {
          Text('Hello World')
        }
      }
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ExampleComponent {
  build() {
    Text('Hello World')
  }
}
```

### Rationale

移除无属性的嵌套 Column 容器，直接在最外层的 Column 中添加 Text 组件，减少视图的嵌套层级。

## Example 5: `pair_c4f9840accf4bcba`

建议尽量减少视图嵌套层次

### Triggering pattern

```arkts
@Entry
@Component
struct ButtonGroup {
  build() {
    Row() {
      Row() {
        Row() {
          Button('Button 1')
          Button('Button 2')
        }
      }
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ButtonGroup {
  build() {
    Row() {
      Button('Button 1')
      Button('Button 2')
    }
  }
}
```

### Rationale

移除多余的嵌套 Row 容器，减少视图嵌套层级，提高渲染性能。
