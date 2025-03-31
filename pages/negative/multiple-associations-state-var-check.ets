// 多个组件关联同一数据时，建议在组件中使用@Watch装饰器添加更新条件，避免不必要的组件更新。
//
// 通用丢帧场景下，建议优先修改。

@Observed
class UIStyle {
  fontSize: number = 0;
  fontColor: string = '';
  isChecked: boolean = false;
}
@Entry
@Component
struct MultipleAssociationsStateVarReport0 {
  @State uiStyle: UIStyle = new UIStyle();
  private listData: string[] = [];
  aboutToAppear(): void {
    for (let i = 0; i < 10; i++) {
      this.listData.push(`ListItemComponent ${i}`);
    }
  }
  build() {
    Row() {
      Column() {
        CompA({item: '1', index: 1, subStyle: this.uiStyle})
        CompB({item: '2', index: 2, subStyle: this.uiStyle})
        CompC({item: '3', index: 3, subStyle: this.uiStyle})
        Text('change state var')
          .onClick(()=>{
            this.uiStyle.fontSize = 20;
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}
@Component
struct CompA {
  @Prop item: string;
  @Prop index: number;
  @Link subStyle: UIStyle;
  private sizeFont: number = 50;
  isRender(): number {
    console.info(`CompA ${this.index} Text is rendered`);
    return this.sizeFont;
  }
  build() {
    Column() {
      Text(this.item)
        .fontSize(this.isRender())
        .fontSize(this.subStyle.fontSize)
      Text('abc')
    }
  }
}
@Component
struct CompB {
  @Prop item: string;
  @Prop index: number;
  @Link subStyle: UIStyle;
  private sizeFont: number = 50;
  isRender(): number {
    console.info(`CompB ${this.index} Text is rendered`);
    return this.sizeFont;
  }
  build() {
    Column() {
      Text(this.item)
        .fontSize(this.isRender())
        .fontColor(this.subStyle.fontColor)
      Text('abc')
    }
  }
}
@Component
struct CompC {
  @Prop item: string;
  @Prop index: number;
  @Link subStyle: UIStyle;
  private sizeFont: number = 50;
  isRender(): number {
    console.info(`CompC ${this.index} Text is rendered`);
    return this.sizeFont;
  }
  build() {
    Column() {
      if (this.subStyle.isChecked) {
        Text('checked')
      } else {
        Text('unchecked')
      }
    }
  }
}