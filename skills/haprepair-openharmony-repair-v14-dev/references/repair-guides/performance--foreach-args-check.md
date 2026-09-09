# @performance/foreach-args-check

Static repair references: 23. These are examples, not patches to copy.
Inspect repository context and checker evidence before adapting an example.

## Example 1: `pair_03aa3b5d7d2b5954`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction : FlexDirection.Column ,
          justifyContent : FlexAlign.SpaceBetween ,
          alignItems : ItemAlign.Start
        }) {
          ForEach(this.apiItems , (item: TestApi , index: number) => {
            this.IngredientItem(item , index)
          })
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_88_100)
    .padding({ top : 5 , right : 10 , left : 10 })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction: FlexDirection.Column,
          justifyContent: FlexAlign.SpaceBetween,
          alignItems: ItemAlign.Start
        }) {
          ForEach(this.apiItems, (item: TestApi, index: number) => {
            this.IngredientItem(item, index)
          }, (item: TestApi, index: number) => item.name)
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_88_100)
    .padding({ top: 5, right: 10, left: 10 })
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用item.name作为key generator的标识符

## Example 2: `pair_05a93e2688f754cc`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      ForEach(this.products, (product: Product) => {
      });
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
    Column() {
      ForEach(this.products, (product: Product) => {
      }, (product: Product) => product.productCode)
    }
  }
}
```

### Rationale

缺少唯一键将导致全部重新渲染，而不是仅更新必要部分。

## Example 3: `pair_1db5d53596665884`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
      Stack() {
        Column() {
          Column({ space: 18 }) {
            if (this.recordingStatus === 0) {
              ForEach(this.sidebarList_1, (sidebar: Resource, index: number) => {
                Image($r('app.media.app_icon'))
                  .width(28)
                  .height(28)
                  .objectFit(ImageFit.Contain)
                if (index === 1) {
                  Divider()
                    .vertical(false)
                    .height(1)
                    .width(22)
                    .color($r('app.color.COLOR_FFFFFF'))
                    .margin({ right: 4 })
                }
              })
            } else if (this.recordingStatus === 2) {
              ForEach(this.sidebarList_2, (sidebar: Resource, index: number) => {
                Image($r('app.media.app_icon'))
                  .width(28)
                  .height(28)
                  .objectFit(ImageFit.Contain)
                if (index === 0) {
                  Divider()
                    .vertical(false)
                    .height(1)
                    .width(22)
                    .color($r('app.color.COLOR_FFFFFF'))
                    .margin({ right: 4 })
                }
                if (index === 1) {
                  Text($r('app.string.Wen'))
                    .textAlign(TextAlign.Center)
                    .fontColor($r('app.color.COLOR_FFFFFF'))
                    .fontSize(22)
                    .fontFamily($r('app.string.Font_family_medium'))
                    .margin({ right: 4 })
                }
              })
            }
          }
          .width('100%')
          .height('50%')
        }
        .width('100%')
        .height('100%')
      }
      .width('100%')
      .height('91%')
      .borderRadius(12)
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
      Stack() {
        Column() {
          Column({ space: 18 }) {
            if (this.recordingStatus === 0) {
              ForEach(this.sidebarList_1, (sidebar: Resource, index: number) => {
                Image($r('app.media.app_icon'))
                  .width(28)
                  .height(28)
                  .objectFit(ImageFit.Contain)
                if (index === 1) {
                  Divider()
                    .vertical(false)
                    .height(1)
                    .width(22)
                    .color($r('app.color.COLOR_FFFFFF'))
                    .margin({ right: 4 })
                }
              }, (sidebar: Resource, index: number) => index)
            } else if (this.recordingStatus === 2) {
              ForEach(this.sidebarList_2, (sidebar: Resource, index: number) => {
                Image($r('app.media.app_icon'))
                  .width(28)
                  .height(28)
                  .objectFit(ImageFit.Contain)
                if (index === 0) {
                  Divider()
                    .vertical(false)
                    .height(1)
                    .width(22)
                    .color($r('app.color.COLOR_FFFFFF'))
                    .margin({ right: 4 })
                }
                if (index === 1) {
                  Text($r('app.string.Wen'))
                    .textAlign(TextAlign.Center)
                    .fontColor($r('app.color.COLOR_FFFFFF'))
                    .fontSize(22)
                    .fontFamily($r('app.string.Font_family_medium'))
                    .margin({ right: 4 })
                }
              }, (sidebar: Resource, index: number) => index)
            }
          }
          .width('100%')
          .height('50%')
        }
        .width('100%')
        .height('100%')
      }
      .width('100%')
      .height('91%')
      .borderRadius(12)

  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用index作为key generator的标识符

## Example 4: `pair_26140586ffd3ab4f`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    HStack() {
      ForEach(this.events, (event, idx) => {
        EventItem(event);
      });
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
    HStack() {
      ForEach(this.events, (event, idx) => {
        EventItem(event);
      }, (event) => event.eventID);
    }
  }
}
```

### Rationale

设置唯一键是提高性能的关键。

## Example 5: `pair_2889afd40200f3f7`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
build() { Row() { ForEach(this.notifications, (note) => { NotificationItem(note); }); } }
```

### Repair pattern

```arkts
build() { Row() { ForEach(this.notifications, (note) => { NotificationItem(note); }, (note) => note.timestamp); } }
```

### Rationale

每个项应有唯一键以提高组件渲染效率。

## Example 6: `pair_28da93f02c0ba065`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Grid() {
      ForEach(this.tasks, (task, index) => {
        TaskItem(task);
      });
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
    Grid() {
      ForEach(this.tasks, (task, index) => {
        TaskItem(task);
      }, (task) => task.id);
    }
  }
}
```

### Rationale

为了提高渲染效率，应该为每个项提供一个唯一的键。

## Example 7: `pair_2a58084d2bf5bab4`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
build() { VStack() { ForEach(this.messages, (msg, index) => { MessageBubble(msg); }); } }
```

### Repair pattern

```arkts
build() { VStack() { ForEach(this.messages, (msg, index) => { MessageBubble(msg); }, (msg) => msg.uuid); } }
```

### Rationale

没有唯一键会导致对整个列表的重新渲染，而不是仅更新变化部分。

## Example 8: `pair_2a971c0bb4029c7b`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
build() { HStack() { ForEach(this.events, (event, idx) => { EventItem(event); }); } }
```

### Repair pattern

```arkts
build() { HStack() { ForEach(this.events, (event, idx) => { EventItem(event); }, (event) => event.eventID); } }
```

### Rationale

设置唯一键是提高性能的关键。

## Example 9: `pair_443d51a6d1a726e0`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
build() { List() { ForEach(this.members, (member, idx) => { MemberItem(member); }); } }
```

### Repair pattern

```arkts
build() { List() { ForEach(this.members, (member, idx) => { MemberItem(member); }, (member) => member.email); } }
```

### Rationale

缺少唯一键会导致不必要的重新渲染。

## Example 10: `pair_69c2aab504e63c90`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List() {
      ForEach(this.members, (member, idx) => {
        MemberItem(member);
      });
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
    List() {
      ForEach(this.members, (member, idx) => {
        MemberItem(member);
      }, (member) => member.email);
    }
    .width(10)
    .height(10)
  }
}
```

### Rationale

缺少唯一键会导致不必要的重新渲染。

## Example 11: `pair_7122a3ce6267f2c5`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Row() {
        //TODO 用常量替换
        Text('仓库列表')
          .align(Alignment.Start)
          .fontColor($r("app.color.text_strong"))
        Blank()
        Image($r("app.media.icon_arrow_right"))
          .height('100%')
          .aspectRatio(1)
          .onClick(() => {
            this.viewModel.onRepos()
          })
      }.width('100%')
      .height(50)
      .padding({ left: 24, right: 24, top: 24 })

      Divider()
        .color($r("app.color.boarder_medium"))
        .margin({ top: 16, bottom: 16 })

      List() {
        ForEach(this.viewModel.repos, (item) => {
          ListItem() {
            RepoItem({ viewModel: $viewModel, repo: item })
          }
        })
      }
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
    Column() {
      Row() {
        //TODO 用常量替换
        Text('仓库列表')
          .align(Alignment.Start)
          .fontColor($r("app.color.text_strong"))
        Blank()
        Image($r("app.media.icon_arrow_right"))
          .height('100%')
          .aspectRatio(1)
          .onClick(() => {
            this.viewModel.onRepos()
          })
      }.width('100%')
      .height(50)
      .padding({ left: 24, right: 24, top: 24 })

      Divider()
        .color($r("app.color.boarder_medium"))
        .margin({ top: 16, bottom: 16 })

      List() {
        ForEach(this.viewModel.repos, (item: Repo) => {
          ListItem() {
            RepoItem({ viewModel: $viewModel, repo: item })
          }
        }, (item: Repo) => item.repoName)
      }
      .width(10)
      .height(10)
    }
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用item.repoName作为key generator的标识符

## Example 12: `pair_7ad0e65207b5b503`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Grid() {
      ForEach(this.comments, (comment, idx) => {
        CommentView(comment);
      });
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
    Grid() {
      ForEach(this.comments, (comment, idx) => {
        CommentView(comment);
      }, (comment) => comment.commentID);
    }
  }
}
```

### Rationale

为每个项提供唯一键可以提高渲染性能。

## Example 13: `pair_7d5c288e2b2fa8af`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      ForEach(this.list, (item) => {
        Text(item.toString())
          .width('90%')
          .height(72)
          .backgroundColor('#fff')
          .borderRadius(15)
          .fontSize(24)
          .textAlign(TextAlign.Center)
          .margin(10)
      }, item => item)
    }.width('100%').backgroundColor('#F1F3F5').padding({ bottom: 10 })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {

    Column() {
      ForEach(this.list, (item: number) => {
        Text(item.toString())
          .width('90%')
          .height(72)
          .backgroundColor('#fff')
          .borderRadius(15)
          .fontSize(24)
          .textAlign(TextAlign.Center)
          .margin(10)
      }, (item: number) => item.toString())
    }.width('100%').backgroundColor('#F1F3F5').padding({ bottom: 10 })
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用index作为key generator的标识符

## Example 14: `pair_7fab2efcd9e0bfa0`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List() {
      ForEach(this.orders, (order, i) => {
        OrderItem(order);
      });
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
struct App {
  build() {
    List() {
      ForEach(this.orders, (order, i) => {
        OrderItem(order);
      }, (order) => order.orderNumber);
    }
    .width(10)
    .height(10)
  }
}
```

### Rationale

没有唯一键会影响渲染效率。

## Example 15: `pair_8cf440aec8e9ffd2`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
build() { Grid() { ForEach(this.tasks, (task, index) => { TaskItem(task); }); } }
```

### Repair pattern

```arkts
build() { Grid() { ForEach(this.tasks, (task, index) => { TaskItem(task); }, (task) => task.id); } }
```

### Rationale

为了提高渲染效率，应该为每个项提供一个唯一的键。

## Example 16: `pair_9c2a3520fbbdab5f`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    VStack() {
      ForEach(this.messages, (msg, index) => {
        MessageBubble(msg);
      });
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
    VStack() {
      ForEach(this.messages, (msg, index) => {
        MessageBubble(msg);
      }, (msg) => msg.uuid);
    }
  }
}
```

### Rationale

没有唯一键会导致对整个列表的重新渲染，而不是仅更新变化部分。

## Example 17: `pair_af824cbd03c1e67d`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction: FlexDirection.Column,
          justifyContent: FlexAlign.SpaceBetween,
          alignItems: ItemAlign.Start
        }) {
          ForEach(this.autoItems, (item: TestAuto, index: number) => {
            this.IngredientItem(item, index)
          })
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_80_100)
    .padding({ top: 5, right: 10, left: 10 })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction : FlexDirection.Column ,
          justifyContent : FlexAlign.SpaceBetween ,
          alignItems : ItemAlign.Start
        }) {
          ForEach(this.autoItems , (item: TestAuto , index: number) => {
            this.IngredientItem(item , index)
          }, (item: TestAuto, index: number) => item.name)
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_80_100)
    .padding({ top : 5 , right : 10 , left : 10 })
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用item.name作为key generator的标识符

## Example 18: `pair_b52690ff6f4328a6`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    VStack() {
      ForEach(this.photos, (photo, index) => {
        PhotoThumbnail(photo);
      });
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
    VStack() {
      ForEach(this.photos, (photo, index) => {
        PhotoThumbnail(photo);
      }, (photo) => photo.uniqueID);
    }
  }
}
```

### Rationale

没有使用唯一键会导致额外的渲染负担。

## Example 19: `pair_b5c253295c178a66`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Stack() {
      ForEach(this.books, (book) => {
        BookItem(book);
      });
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
    Stack() {
      ForEach(this.books, (book) => {
        BookItem(book);
      }, (book) => book.isbn);
    }
  }
}
```

### Rationale

缺少唯一键会导致不必要的渲染操作。

## Example 20: `pair_c2928a5983cc2c06`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct ForeachTest {
  private data: string[] = ['1', '2', '3'];

  build() {
    RelativeContainer() {
      List() {
        ForEach(this.data, (item: string, index: number) => {
          ListItem() {
            Text(item);
          }
        })
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
    .width('100%')
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct ForeachTest {
  private data: string[] = ['1', '2', '3'];

  build() {
    RelativeContainer() {
      List() {
        ForEach(this.data, (item: string, index: number) => {
          ListItem() {
            Text(item);
          }
        }, (item: string, index: number) => item)
      }
      .width('100%')
      .height('100%')
    }
    .height('100%')
    .width('100%')
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表

## Example 21: `pair_f08bfdbcea08b0b3`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Row() {
      ForEach(this.notifications, (note) => {
        NotificationItem(note);
      });
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
    Row() {
      ForEach(this.notifications, (note) => {
        NotificationItem(note);
      }, (note) => note.timestamp);
    }
  }
}
```

### Rationale

每个项应有唯一键以提高组件渲染效率。

## Example 22: `pair_fc2983a304ecd0b2`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List({ space: 16 }) {
      ForEach(this.viewModel.issues, (item) => {
        ListItem() {
          IssueItem({ viewModel: this.viewModel, issue: item })
        }
      })
    }.padding(20)
    .width('100%')
    .height('100%')
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    List({ space: 16 }) {
      ForEach(this.viewModel.issues, (item: Issue) => {
        ListItem() {
          IssueItem({ viewModel: this.viewModel, issue: item })
        }
      }, (item: Issue) => item.id)
    }.padding(20)
    .width(10)
    .height(10)
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用item.id作为key generator的标识符

## Example 23: `pair_fc41d97cdb612dc2`

建议在ForEach参数中设置keyGenerator

### Triggering pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction: FlexDirection.Column,
          justifyContent: FlexAlign.SpaceBetween,
          alignItems: ItemAlign.Start
        }) {
          ForEach(this.scenarioItems, (item: TestScenario, index: number) => {
            this.IngredientItem(item, index)
          })
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_80_100)
    .padding({ top: 5, right: 10, left: 10 })
  }
}
```

### Repair pattern

```arkts
@Entry
@Component
struct App {
  build() {
    Column() {
      Scroll() {
        Flex({
          direction: FlexDirection.Column,
          justifyContent: FlexAlign.SpaceBetween,
          alignItems: ItemAlign.Start
        }) {
          ForEach(this.scenarioItems, (item: TestScenario, index: number) => {
            this.IngredientItem(item, index)
          }, (item: TestScenario, index: number) => item.name)
        }
      }
      .scrollBarWidth(20)
    }
    .height(ConfigData.WH_80_100)
    .padding({ top: 5, right: 10, left: 10 })
  }
}
```

### Rationale

通过为每个项设置唯一的键，框架可以更高效地识别哪些项发生了变化，从而只重新渲染那些实际需要更新的部分，而不是整个列表。这里使用item.name作为key generator的标识符
