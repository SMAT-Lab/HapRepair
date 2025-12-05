import { CommonNavigation } from './CommonNavigation';

export interface ClickItem {
  name: string;
  callback: () => void;
  id?: string;
}

@Component
struct SheetComponent {
  pathStack: NavPathStack = new NavPathStack();
  clickItem: Array<ClickItem> = [{
    name: '关闭半模态',
    callback: () => {
      AppStorage.setOrCreate('isSheetShow', false);
    },
    id: 'closeSheetId'
  }]

  @Builder
  pageBuilder(name: string) {
    if (name == "pageOne") {
      NavDestination() {
      }.title('pageOne')
      .onActive((reason: NavDestinationActiveReason) => {
        console.log("trigger sheet pageOne onActive: " + reason);
      }).onInactive((reason: NavDestinationActiveReason) => {
        console.log("trigger sheet pageOne onInactive: " + reason);
      })
    }
  }
  build() {
    Column() {
      Scroll() {
        Column() {
          ForEach(this.clickItem, (item: ClickItem) => {
            Button(item.name).width(300).height(30).margin(10).onClick(item.callback).id(item.id)
          })
        }
      }.width('100%').height('50%')
      Navigation(this.pathStack) {
      }.title('Sheet Navigation')
      .navDestination(this.pageBuilder)
    }.backgroundColor(Color.Yellow)
  }
}

@Entry
@Component
struct BasicSheet01 {
  pathStack: NavPathStack = new NavPathStack()
  @StorageLink('isShow') isShow: boolean = false;
  @StorageLink('isSheetShow') isSheetShow: boolean = false;
  @State mode: SheetMode = SheetMode.OVERLAY
  clickItem: Array<ClickItem> = [
    {
      name: 'push pageOne',
      callback: () => {
        this.pathStack.pushPath({name: 'pageOne'})
      },
      id: 'sheetPushId'
    }, {
      name: '开启半模态',
      callback: () => {
        this.isSheetShow = true;
        this.mode = SheetMode.OVERLAY
      },
      id: 'openSheetId'
    }
  ]

  @Builder
  SheetBuilder() {
    SheetComponent()
  }

  build() {
    Column() {
      Scroll() {
        Column() {
          ForEach(this.clickItem, (item: ClickItem) => {
            Button(item.name).width(300).height(30).margin(10).onClick(item.callback).id(item.id)
          })
        }
      }.height('50%')

      CommonNavigation({pathStack: this.pathStack})
        .bindSheet(this.isSheetShow, this.SheetBuilder, {mode: this.mode})
    }
  }
}