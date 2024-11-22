

@Reusable
@Component
struct ItemComponent {
  @State value: string = '';
  @State result: number = 0;

  aboutToReuse(params: Record<string, Object>): void {
    this.value = params.value as string;
    this.result = params.result as number;
  }

  build() {
    Column() {
      Text('Item: ' + this.value)
        .fontSize(24)
      Text('Result: ' + this.result)
        .fontSize(24)
    }
  }
}

@Entry
@Component
struct ParentComponent {
  private items: string[] = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'];

  // 模拟一个耗时的计算函数
  expensiveOperation(): number {
    let sum = 0;
    for (let i = 0; i < 10000; i++) {
      sum += i;
    }
    return sum;
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.items, (item: string) => {
          ListItem() {
            // 避免在传入组件时调用函数
            ItemComponent({ value: item, result: this.expensiveOperation() })
          }
          .width('100%')
          .height(100)
        }, (item: string) => item)
      }
      .cachedCount(2)
      .height('100%')
      .width('100%')
    }
  }
}