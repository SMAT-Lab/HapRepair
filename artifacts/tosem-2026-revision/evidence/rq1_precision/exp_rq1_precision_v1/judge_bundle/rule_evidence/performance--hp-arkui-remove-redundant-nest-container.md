# @performance/hp-arkui-remove-redundant-nest-container

Static repair references: 5. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_01723de3778e1fa2`

避免冗余的嵌套

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent {
    @State children: Number[] = Array.from(Array<number>(900), (v, k) => k);
    
    build() {
      Scroll() {
      Grid() {
        ForEach(this.children, (item: Number[]) => {
          GridItem() {
            // 冗余Stack
            Stack() {  
              Stack() {  
                Stack() {  
                  Text(item.toString())  
                }.size({ width: "100%"})  
              }.backgroundColor(Color.Yellow)  
            }.backgroundColor(Color.Pink)  
          }  
        }, (item: string) => item)  
      }  
      .columnsTemplate('1fr 1fr 1fr 1fr')  
      .columnsGap(0)  
      .rowsGap(0)  
      .size({ width: "100%", height: "100%" })  
    }  
  }  
}
```

### Repair pattern

```arkts
@Entry  
@Component  
struct MyComponent {  
  @State children: Number[] = Array.from(Array<number>(900), (v, k) => k);  
  
  build() {  
    Scroll() {  
      Grid() {  
        ForEach(this.children, (item: Number[]) => {  
          GridItem() {  
            Text(item.toString())  
          }.backgroundColor(Color.Yellow)  
        }, (item: string) => item)  
      }  
      .columnsTemplate('1fr 1fr 1fr 1fr')  
      .columnsGap(0)  
      .rowsGap(0)  
      .size({ width: "100%", height: "100%" })  
    }  
  }  
}
```

### Rationale

移除冗余Stack

## Example 2: `pair_2743ab9e527d04ab`

避免冗余的嵌套

### Triggering pattern

```arkts
@Entry
@Component
struct ComplexLayout {
  build() {
    Stack() {
      Stack() {
        Stack() {
          Rectangle()
            .width(100)
            .height(100)
            .backgroundColor(Color.Red)
        }
        .padding(10)
      }
      .borderRadius(10)
    }
    .shadow({ color: Color.Black, radius: 5, offset: { x: 0, y: 2 } })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ComplexLayout {
  build() {
    Stack() {
      Rectangle()
        .width(100)
        .height(100)
        .backgroundColor(Color.Red)
        .padding(10)
        .borderRadius(10)
    }
    .shadow({ color: Color.Black, radius: 5, offset: { x: 0, y: 2 } })
  }
}
```

### Rationale

将属性如 padding、borderRadius 等应用到实际的内容组件上，移除多余的 Stack 嵌套。

## Example 3: `pair_28eae663c02bb482`

避免冗余的嵌套

### Triggering pattern

```arkts
@Entry
@Component
struct MyComponent{
  @State number: Number[] = Array.from(Array<number>(1000), (val, i) => i);
  scroller: Scroller = new Scroller()
  build() {
    Column() {
      Grid(this.scroller) {
        ForEach(this.number, (item: number) => {
          GridItem() {
            Flex() {
              Flex() {
                Flex() {
                  Text(item.toString())
                    .fontSize(16)
                    .backgroundColor(0xF9CF93)
                    .width('100%')
                    .height(80)
                    .textAlign(TextAlign.Center)
                    .border({width:1})
                }
              }
            }
          }
        }, (item:string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
      .columnsGap(0)
      .rowsGap(0)
      .size({ width: "100%", height: "100%" })
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct MyComponent{
  @State number: Number[] = Array.from(Array<number>(1000), (val, i) => i);
  scroller: Scroller = new Scroller()
  build() {
    Column() {
      Grid(this.scroller) {
        ForEach(this.number, (item: number) => {
          GridItem() {
            Text(item.toString())
              .fontSize(16)
              .backgroundColor(0xF9CF93)
              .width('100%')
              .height(80)
              .textAlign(TextAlign.Center)
              .border({width:1})
          }
        }, (item:string) => item)
      }
      .columnsTemplate('1fr 1fr 1fr 1fr 1fr')
      .columnsGap(0)
      .rowsGap(0)
      .size({ width: "100%", height: "100%" })
    }
  }
}
```

### Rationale

嵌套了三层Flex()，但是这三层视图没有任何其它属性，均是可有可无的冗余视图，因此直接移除

## Example 4: `pair_294e6ccce68430df`

避免冗余的嵌套

### Triggering pattern

```arkts
@Entry
@Component
struct NestedButtons {
  build() {
    Column() {
      Row() {
        Flex() {
          Flex() {
            Button('Button A')
              .fontSize(16)
          }
          .justifyContent(FlexAlign.Center)
        }
        .alignItems(FlexAlign.Center)
      }
      .backgroundColor(Color.LightGray)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct NestedButtons {
  build() {
      Flex() {
        Button('Button A')
          .fontSize(16)
      }
      .justifyContent(FlexAlign.Center)
      .alignItems(FlexAlign.Center)
      .backgroundColor(Color.LightGray)
  }
}
```

### Rationale

将子级 Flex 容器的属性上移到父级 Flex 容器，减少不必要的嵌套层级。

## Example 5: `pair_c02c255c114f9faa`

避免冗余的嵌套

### Triggering pattern

```arkts
@Entry
@Component
struct CardList {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Column() {
            Column() {
              Column() {
                Text('Card ' + item)
              }
              .padding(10)
            }
            .backgroundColor(Color.White)
          }
          .borderRadius(8)
          .margin(5)
        }
      }, (item:number)=>item.toString())
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
struct CardList {
  build() {
    List() {
      ForEach([1, 2, 3], (item) => {
        ListItem() {
          Column() {
            Text('Card ' + item)
              .padding(10)
          }
          .backgroundColor(Color.White)
          .borderRadius(8)
          .margin(5)
        }
      }, (item: number)=> item.toString())
    }
    .height('100%')
    .width('100%')
    .cachedCount(2)
  }
}
```

### Rationale

将嵌套的 Column 容器的属性（如 padding、backgroundColor）上移，合并到一个 Column 中，移除冗余的嵌套。
