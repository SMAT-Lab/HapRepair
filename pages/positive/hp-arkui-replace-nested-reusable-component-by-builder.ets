// 源码文件，请以工程实际为准
import { MyDataSource } from './MyDataSource';

@Entry
@Component
struct MyComponent{
  private data: MyDataSource = new MyDataSource();

  aboutToAppear(): void {
    for (let index = 0; index < 30; index++) {
      this.data.pushData(index.toString())
    }
  }

  build() {
    Column() {
      List() {
        LazyForEach(this.data, (item: string) => {
          ListItem() {
            //  正例
            ChildComponent({ desc: item })
          }
        }, (item: string) => item)
      }
      .height('100%')
      .width('100%')
    }
    .width('100%')
  }
}

// 正例 使用组件复用
@Reusable
@Component
struct ChildComponent {
  @State desc: string = '';

  aboutToReuse(params: Record<string, Object>): void {
    this.desc = params.desc as string;
  }

  build() {
    Column() {
      // 使用@Builder，可以减少自定义组件创建和渲染的耗时
      ChildComponentBuilder({ paramA: this.desc })
    }
    .width('100%')
  }
}

class Temp {
  paramA: string = '';
}

@Builder
function ChildComponentBuilder($$: Temp) {
  Column() {
    // 此处使用`${}`来进行按引用传递，让@Builder感知到数据变化，进行UI刷新
    Text(`子组件 + ${$$.paramA}`)
      .fontSize(30)
      .fontWeight(30)
  }
  .width('100%')
}