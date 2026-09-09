# @performance/waterflow-data-preload-check

Static repair references: 20. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_02604667a474d8d0`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .width('100%')
    .height(this.itemHeightArray[item % 25])
    .backgroundColor(this.colors[item % 6])
  }, (item: string) => item)
}
.onReachEnd(() => {
  console.info("Scroll end detected")
  setTimeout(() => {
    for (let i = 0; i < 110; i++) {
      this.dataSource.expandList()
    }
  }, 1500)
})
```

### Repair pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .onAppear(() => {
      if (item + 5 == this.dataSource.totalCount()) {
        for (let i = 0; i < 110; i++) {
          this.dataSource.expandList()
        }
      }
    })
    .width('100%')
    .height(this.itemHeightArray[item % 25])
    .backgroundColor(this.colors[item % 6])
  }, (item: string) => item)
}
```

### Rationale

在 onReachEnd 中处理大量数据加载可能会造成卡顿。通过检测 FlowItem 滑入视图并提前加载数据，可提升用户体验。

## Example 2: `pair_20a0304e58b5d063`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .width('100%')
    .height(this.itemHeightArray[item % 60])
    .backgroundColor(this.colors[item % 3])
  }, (item: string) => item)
}
.onReachEnd(() => {
  console.info("Load end")
  for (let i = 0; i < 70; i++) {
    this.dataSource.addBatchItem()
  }
})
```

### Repair pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .onAppear(() => {
      if (item + 15 == this.dataSource.totalCount()) {
        for (let i = 0; i < 70; i++) {
          this.dataSource.addBatchItem()
        }
      }
    })
    .width('100%')
    .height(this.itemHeightArray[item % 60])
    .backgroundColor(this.colors[item % 3])
  }, (item: string) => item)
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能导致卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 3: `pair_356a49d5284e7e17`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      WaterFlow() {
        LazyForEach(this.elements, (item: number) => {
          FlowItem() {
            ComponentFlow({ item: item })
          }
          .width('100%')
          .height(this.heightMap[item % 35])
          .backgroundColor(this.colorMap[item % 4])
        }, (item: string) => item)
      }
      .cachedCount(2)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.elements, (item: number) => {
        FlowItem() {
          ComponentFlow({ item: item })
        }
        .onAppear(() => {
          if (item + 3 == this.elements.totalCount()) {
            for (let i = 0; i < 20; i++) {
              this.elements.addMore()
            }
          }
        })
        .width('100%')
        .height(this.heightMap[item % 35])
        .backgroundColor(this.colorMap[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 动态增加条目可能引起界面不流畅。应在 FlowItem 入口时提前加载来平滑过渡。

## Example 4: `pair_3ed1607f3d014671`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 90])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Appending items")
      for (let i = 0; i < 90; i++) {
        this.dataSource.increaseCollection()
      }
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 22 == this.dataSource.totalCount()) {
            for (let i = 0; i < 90; i++) {
              this.dataSource.increaseCollection()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 90])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 中批量添加数据可能导致滚动的中断。通过更早触发数据加载可优化性能。

## Example 5: `pair_434fbee1baa3f22b`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .width('100%')
    .height(this.itemHeightArray[item % 90])
    .backgroundColor(this.colors[item % 5])
  }, (item: string) => item)
}
.onReachEnd(() => {
  console.info("Appending items")
  for (let i = 0; i < 90; i++) {
    this.dataSource.increaseCollection()
  }
})
```

### Repair pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .onAppear(() => {
      if (item + 22 == this.dataSource.totalCount()) {
        for (let i = 0; i < 90; i++) {
          this.dataSource.increaseCollection()
        }
      }
    })
    .width('100%')
    .height(this.itemHeightArray[item % 90])
    .backgroundColor(this.colors[item % 5])
  }, (item: string) => item)
}
```

### Rationale

在 onReachEnd 中批量添加数据可能导致滚动的中断。通过更早触发数据加载可优化性能。

## Example 6: `pair_644cdfe7450e091b`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 20])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Loading more items")
      for (let i = 0; i < 20; i++) {
        this.dataSource.addMoreItems()
      }
    })

  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 10 == this.dataSource.totalCount()) {
            for (let i = 0; i < 20; i++) {
              this.dataSource.addMoreItems()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 20])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能会导致滚动到底部时的卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 7: `pair_6ab881af8c430acf`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 25])
        .backgroundColor(this.colors[item % 6])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Scroll end detected")
      setTimeout(() => {
        for (let i = 0; i < 110; i++) {
          this.dataSource.expandList()
        }
      }, 1500)
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 5 == this.dataSource.totalCount()) {
            for (let i = 0; i < 110; i++) {
              this.dataSource.expandList()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 25])
        .backgroundColor(this.colors[item % 6])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 中处理大量数据加载可能会造成卡顿。通过检测 FlowItem 滑入视图并提前加载数据，可提升用户体验。

## Example 8: `pair_6d4a17a0e9732dff`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      WaterFlow() {
        LazyForEach(this.sourceData, (item: number) => {
          FlowItem() {
            CustomFlowItem({ item: item })
          }
          .width('100%')
          .height(this.sizeArray[item % 50])
          .backgroundColor(this.hues[item % 2])
        }, (item: string) => item)
      }
      .cachedCount(2)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.sourceData, (item: number) => {
        FlowItem() {
          CustomFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 8 == this.sourceData.totalCount()) {
            for (let i = 0; i < 40; i++) {
              this.sourceData.updateCollection()
            }
          }
        })
        .width('100%')
        .height(this.sizeArray[item % 50])
        .backgroundColor(this.hues[item % 2])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中同步增长数据会阻塞。改为在 FlowItem 即将出现时预加载。

## Example 9: `pair_6f34b61313fe4ca1`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 100])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("onReachEnd")
      setTimeout(() => {
        for (let i = 0; i < 100; i++) {
          this.datasource.AddLastItem()
        }
      }, 1000)
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {

    LazyForEach(this.dataSource, (item: number) => {
      FlowItem() {
        ReusableFlowItem({ item: item })
      }
      .onAppear(() => {
        if (item + 20 == this.dataSource.totalCount()) {
          for (let i = 0; i < 100; i++) {
            this.dataSource.addLastItem()
          }
        }
      })
      .width('100%')
      .height(this.itemHeightArray[item % 100])
      .backgroundColor(this.colors[item % 5])
    }, (item: string) => item)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能会导致滚动到底部时的卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 10: `pair_76b909da0905a724`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .width('100%')
    .height(this.itemHeightArray[item % 30])
    .backgroundColor(this.colors[item % 3])
  }, (item: string) => item)
}
.onReachEnd(() => {
  console.info("Fetching more data")
  setTimeout(() => {
    for (let i = 0; i < 60; i++) {
      this.dataSource.fetchNewData()
    }
  }, 700)
})
```

### Repair pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .onAppear(() => {
      if (item + 18 == this.dataSource.totalCount()) {
        for (let i = 0; i < 60; i++) {
          this.dataSource.fetchNewData()
        }
      }
    })
    .width('100%')
    .height(this.itemHeightArray[item % 30])
    .backgroundColor(this.colors[item % 3])
  }, (item: string) => item)
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能导致滞后。通过更早检测FlowItem的出现并预加载数据可避免此问题。

## Example 11: `pair_9def5d7d32d9845d`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      WaterFlow() {
        LazyForEach(this.listData, (item: number) => {
          FlowItem() {
            CustomReusableItem({ item: item })
          }
          .width('100%')
          .height(this.dimensionArray[item % 45])
          .backgroundColor(this.colorPatterns[item % 3])
        }, (item: string) => item)
      }
      .cachedCount(2)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.listData, (item: number) => {
        FlowItem() {
          CustomReusableItem({ item: item })
        }
        .onAppear(() => {
          if (item + 13 == this.listData.totalCount()) {
            for (let i = 0; i < 30; i++) {
              this.listData.extendItems()
            }
          }
        })
        .width('100%')
        .height(this.dimensionArray[item % 45])
        .backgroundColor(this.colorPatterns[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 使用异步函数加载数据将延后用户体验。通过 FlowItem 出现时检测进行提前加载优化效果。

## Example 12: `pair_a784ce33d47cf402`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .width('100%')
    .height(this.itemHeightArray[item % 70])
    .backgroundColor(this.colors[item % 4])
  }, (item: string) => item)
}
.onReachEnd(() => {
  console.info("More items coming")
  setTimeout(() => {
    for (let i = 0; i < 80; i++) {
      this.dataSource.loadItems()
    }
  }, 1200)
})
```

### Repair pattern

```arkts
WaterFlow() {
  LazyForEach(this.dataSource, (item: number) => {
    FlowItem() {
      ReusableFlowItem({ item: item })
    }
    .onAppear(() => {
      if (item + 12 == this.dataSource.totalCount()) {
        for (let i = 0; i < 80; i++) {
          this.dataSource.loadItems()
        }
      }
    })
    .width('100%')
    .height(this.itemHeightArray[item % 70])
    .backgroundColor(this.colors[item % 4])
  }, (item: string) => item)
}
```

### Rationale

在 onReachEnd 中异步加载过多数据可能导致反应延迟。通过在 FlowItem 显示时预加载来改善性能。

## Example 13: `pair_b741d4d97fe67207`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App{
  build() {
    Row() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 30])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 18 == this.dataSource.totalCount()) {
            for (let i = 0; i < 60; i++) {
              this.dataSource.fetchNewData()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 30])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能导致滞后。通过更早检测FlowItem的出现并预加载数据可避免此问题。

## Example 14: `pair_b990a36426ea21c2`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 100])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 20 == this.dataSource.totalCount()) {
            for (let i = 0; i < 50; i++) {
              this.dataSource.addNewItem()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 100])
        .backgroundColor(this.colors[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能会导致滚动到底部时的卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 15: `pair_d37ed05385640b80`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 80])
        .backgroundColor(this.colors[item % 2])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("End reached")
      setTimeout(() => {
        for (let i = 0; i < 40; i++) {
          this.dataSource.addItems()
        }
      }, 1000)
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 25 == this.dataSource.totalCount()) {
            for (let i = 0; i < 40; i++) {
              this.dataSource.addItems()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 80])
        .backgroundColor(this.colors[item % 2])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能会导致滚动到底部时的卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 16: `pair_e278eb87d77b69a0`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 70])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("More items coming")
      setTimeout(() => {
        for (let i = 0; i < 80; i++) {
          this.dataSource.loadItems()
        }
      }, 1200)
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 12 == this.dataSource.totalCount()) {
            for (let i = 0; i < 80; i++) {
              this.dataSource.loadItems()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 70])
        .backgroundColor(this.colors[item % 4])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 中异步加载过多数据可能导致反应延迟。通过在 FlowItem 显示时预加载来改善性能。

## Example 17: `pair_e54e67f9754ee91f`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 50])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("End of List")
      for (let i = 0; i < 30; i++) {
        this.dataSource.appendItem()
      }
    })

  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 15 == this.dataSource.totalCount()) {
            for (let i = 0; i < 30; i++) {
              this.dataSource.appendItem()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 50])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能会导致滚动到底部时的卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 18: `pair_eccc927d089a0a92`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .width('100%')
        .height(this.itemHeightArray[item % 60])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
    .onReachEnd(() => {
      console.info("Load end")
      for (let i = 0; i < 70; i++) {
        this.dataSource.addBatchItem()
      }
    })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.dataSource, (item: number) => {
        FlowItem() {
          ReusableFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 15 == this.dataSource.totalCount()) {
            for (let i = 0; i < 70; i++) {
              this.dataSource.addBatchItem()
            }
          }
        })
        .width('100%')
        .height(this.itemHeightArray[item % 60])
        .backgroundColor(this.colors[item % 3])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 事件中加载更多数据可能导致卡顿。通过提前检测 FlowItem 的出现并预加载数据，可以避免这种情况。

## Example 19: `pair_ed5d64d3f4b394cf`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App{
  build(){
    Row() {
      LazyForEach(this.itemSource, (item: number) => {
        FlowItem() {
          BaseFlowItem({ item: item })
        }
        .width('100%')
        .height(this.configHeight[item % 40])
        .backgroundColor(this.palette[item % 5])
      }, (item: string) => item)
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    WaterFlow() {
      LazyForEach(this.itemSource, (item: number) => {
        FlowItem() {
          BaseFlowItem({ item: item })
        }
        .onAppear(() => {
          if (item + 9 == this.itemSource.totalCount()) {
            for (let i = 0; i < 60; i++) {
              this.itemSource.bringMoreData()
            }
          }
        })
        .width('100%')
        .height(this.configHeight[item % 40])
        .backgroundColor(this.palette[item % 5])
      }, (item: string) => item)
    }
    .cachedCount(2)
  }
}
```

### Rationale

在 onReachEnd 使用延迟函数进行数据装载可能减慢响应。应通过检测 FlowItem 出现提前载入数据来提高效率。

## Example 20: `pair_f895bb257cdf4979`

建议对waterflow子组件进行数据预加载。

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column({ space: 2 }) {
      WaterFlow() {
        LazyForEach(this.dataSource, (item: number) => {
          FlowItem() {
            ReusableFlowItem({ item: item })
          }
          .width('100%')
          .height(this.itemHeightArray[item % 20])
          .backgroundColor(this.colors[item % 2])
        }, (item: string) => item)
      }
      .cachedCount(2)
      .onReachEnd(() => {
        console.info("Adding more data")
        setTimeout(() => {
          for (let i = 0; i < 50; i++) {
            this.dataSource.addLastItem()
          }
        }, 800)
      })
      .columnsTemplate("1fr 1fr")
      .columnsGap(10)
      .rowsGap(5)
      .backgroundColor(0xFAEEE0)
      .width('100%')
      .height('100%')
    }
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column({ space: 2 }) {
      WaterFlow() {
        LazyForEach(this.dataSource, (item: number) => {
          FlowItem() {
            ReusableFlowItem({ item: item })
          }
          .onAppear(() => {
            if (item + 10 == this.dataSource.totalCount()) {
              for (let i = 0; i < 50; i++) {
                this.dataSource.addLastItem()
              }
            }
          })
          .width('100%')
          .height(this.itemHeightArray[item % 20])
          .backgroundColor(this.colors[item % 2])
        }, (item: string) => item)
      }
      .cachedCount(2)
      .columnsTemplate("1fr 1fr")
      .columnsGap(10)
      .rowsGap(5)
      .backgroundColor(0xFAEEE0)
      .width('100%')
      .height('100%')
    }
  }
}
```

### Rationale

在 onReachEnd 时异步加载数据导致延迟。应在 FlowItem 出现时预加载提高性能。
