
        LazyForEach(this.data, (item: Item) => {
          ListItem() {
            ChildComponent({ desc: item.id, sum: 0, avg: 0 })
          }
          .width('100%')
          .height('10%')
          .border({ width: 1 })
          .borderStyle(BorderStyle.Dashed)
        }, (item: Item) => JSON.stringify(item))