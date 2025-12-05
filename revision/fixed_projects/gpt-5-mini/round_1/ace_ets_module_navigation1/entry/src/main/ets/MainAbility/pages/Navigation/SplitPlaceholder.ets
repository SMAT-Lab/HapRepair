
import { ComponentContent, SymbolGlyphModifier } from '@kit.ArkUI';

@Builder function PlaceholderPage(){
  Column({space:10}) {
    Text('分栏模式占位页面').id('text1Id')
      .fontSize(28)
  }.width('100%')
  .height('100%')
  .backgroundColor(Color.Green)

}

@Builder function PlaceholderPage2(){
  Column({space:10}) {
    Text('分栏模式占位页面').id('text2Id')
      .fontSize(28)
  }.width('100%')
  .height("100%")
  .backgroundColor(Color.Pink)
  .expandSafeArea([SafeAreaType.SYSTEM],[SafeAreaEdge.START])
}

@Builder function PlaceholderPage3() {

}

@Entry
@Component
struct SplitPlaceholder {
  @Provide('navPathStack') navPathStack: NavPathStack = new NavPathStack();
  placeholder = new ComponentContent(this.getUIContext(),wrapBuilder(PlaceholderPage))
  placeholder2 = new ComponentContent(this.getUIContext(),wrapBuilder(PlaceholderPage2))
  placeholder3 = new ComponentContent(this.getUIContext(),wrapBuilder(PlaceholderPage3))

  @State menuItems: Array<NavigationMenuItem> = [
    {
      value: 'menuItem2',
      symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus')).fontColor([Color.Red, Color.Green])
        .renderingStrategy(SymbolRenderingStrategy.MULTIPLE_COLOR),
    },
    {
      value: 'menuItem3',
      symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_lungs')),
    },
  ]
  @State toolItems: Array<ToolbarItem> = [
    {
      value: 'toolItem1',
      symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_lungs')),
      status: ToolbarItemStatus.ACTIVE,
      activeSymbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus')).fontColor([Color.Red,
        Color.Green]).renderingStrategy(SymbolRenderingStrategy.MULTIPLE_COLOR),
      action: () => {
      }
    },
    {
      value: 'toolItem2',
      symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_star')),
      status: ToolbarItemStatus.ACTIVE,
      action: () => {
      }
    },
    {
      value: 'toolItem3',
      symbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_star')),
      status: ToolbarItemStatus.ACTIVE,
      activeSymbolIcon: new SymbolGlyphModifier($r('sys.symbol.ohos_lungs')),
      action: () => {
      }
    }
  ]
  @State splitPlaceholder:(ComponentContent<Object>|undefined|null) [] = [this.placeholder, this.placeholder2, this.placeholder3, null, undefined]
  @State splitPlaceholderIndex:number = 0
  @State navBarPosition:NavBarPosition[] =[NavBarPosition.Start,NavBarPosition.End]
  // navBarPositionIndex 未发生变化，改为普通变量
  navBarPositionIndex:number = 0
  @State hideNavBar:boolean[] =[false, true]
  // hideNavBarIndex 未发生变化，改为普通变量
  hideNavBarIndex:number = 0
  // SafeAreaEdgeIndex 未发生变化，改为普通变量
  SafeAreaEdgeIndex:number =0
  @State enableDragBar:boolean[]=[true, false]
  // enableDragBarIndex 未发生变化，改为普通变量
  enableDragBarIndex:number =0
  @State enableMode:boolean []=[false, true]
  // enableModeIndex 未发生变化，改为普通变量
  enableModeIndex:number = 0

  build() {
    Navigation(this.navPathStack) {
      Scroll() {
        Column({space:10}) {
          Button('splitPlaceholder', { stateEffect: true, type: ButtonType.Capsule })
            .width('80%')
            .height('10%')
            .id('btn1')
            .backgroundColor(Color.Blue)
            .onClick(() => {
                this.splitPlaceholderIndex = 0
            })

          Button('splitPlaceholder2', { stateEffect: true, type: ButtonType.Capsule })
            .width('80%')
            .height('10%')
            .id('btn2')
            .backgroundColor(Color.Blue)
            .onClick(() => {
              this.splitPlaceholderIndex = 1
            })

          Button('splitPlaceholder空节点', { stateEffect: true, type: ButtonType.Capsule })
            .width('80%')
            .height('10%')
            .id('btn3')
            .backgroundColor(Color.Blue)
            .onClick(() => {
              this.splitPlaceholderIndex = 2
            })

          Button('null splitPlaceholder', { stateEffect: true, type: ButtonType.Capsule })
            .width('80%')
            .height('10%')
            .id('btn4')
            .backgroundColor(Color.Blue)
            .onClick(() => {
              this.splitPlaceholderIndex = 3
            })

          Button('undefined splitPlaceholder', { stateEffect: true, type: ButtonType.Capsule })
            .width('80%')
            .height('10%')
            .id('btn5')
            .backgroundColor(Color.Blue)
            .onClick(() => {
              this.splitPlaceholderIndex = 4
            })

        }
      }
    }
    .enableModeChangeAnimation(this.enableMode[this.enableModeIndex])
    .enableDragBar(this.enableDragBar[this.enableDragBarIndex])
    .hideNavBar(this.hideNavBar[this.hideNavBarIndex])
    .navBarPosition(this.navBarPosition[this.navBarPositionIndex])
    .backButtonIcon(new SymbolGlyphModifier($r('sys.symbol.ohos_wifi')))
    .titleMode(NavigationTitleMode.Mini)
    .menus(this.menuItems)
    .toolbarConfiguration(this.toolItems)
    .title('一级页面')
    .splitPlaceholder(this.splitPlaceholder[this.splitPlaceholderIndex])
    .mode(NavigationMode.Split)
    .navBarWidthRange(['50%',undefined])
  }
}