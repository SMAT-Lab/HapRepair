// 建议使用@Builder替代嵌套的自定义组件。
//
// 通用丢帧场景下，建议优先修改。

// 源码文件，请以工程实际为准
import { MyDataSource } from './m/MyDataSource';

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource();

  aboutToAppear(): void {
    for (let index = 0; index < 30; index++) {
      this.data.pushData(index)
    }
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.data, (item: string) => {
          ListItem() {
            //反例 使用自定义组件
            ComponentA({ desc: item })
          }
        }, (item: string) => item)
      }
      .height('100%')
      .width('100%')
    }
  }
}

@Reusable
@Component
struct ComponentA {
  @State desc: string = '';

  aboutToReuse(params: ESObject): void {
    this.desc = params.desc as string;
  }

  build() {
    // 在复用组件中嵌套使用自定义组件
    ComponentB({ desc: this.desc })
  }
}



@Component
struct ComponentB {
  @State desc: string = '';

  // 嵌套的组件中也需要实现aboutToReuse来进行UI的刷新
  aboutToReuse(params: ESObject): void {
    this.desc = params.desc as string;
  }

  build() {
    Column() {
      Text('子组件' + this.desc)
        .fontSize(30)
        .fontWeight(30)
    }
    .width('100%')
  }
}