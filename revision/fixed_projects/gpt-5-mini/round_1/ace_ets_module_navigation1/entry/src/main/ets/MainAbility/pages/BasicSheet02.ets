

import { CommonNavigation } from './CommonNavigation';

export interface ClickItem {
  name: string;
  callback: () => void;
  id?: string;
}


@Component
struct ModalComponent {
  pathStack: NavPathStack = new NavPathStack()
  clickItem: Array<ClickItem> = [
    {
      name: '关闭全模态',
      callback: () => {
        AppStorage.setOrCreate('isShow', false);
      },
      id: 'closeAllSheetId'
    }
  ]

  @Builder
  pageBuilder(name: string) {
    if (name == "pageOne") {
      NavDestination() {
      }.title('pageOne')
      .onActive((reason: NavDestinationActiveReason) => {
        console.log('modal pageOne onActive: ' + reason);
      }).onInactive((reason: NavDestinationActiveReason) => {
        console.log('modal pageOne onInactive: ' + reason);
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
      }.height('50%')
      Navigation(this.pathStack) {
      }.title('Modal Navigation').width('100%').navDestination(this.pageBuilder)
    }.backgroundColor(Color.Yellow.toString())
  }
}

@Entry
@Component
struct BasicSheet02 {
  pathStack: NavPathStack = new NavPathStack()
  @StorageLink('isShow') isShow: boolean = false;
  @StorageLink('isSheetShow') isSheetShow: boolean = false;
  mode: SheetMode = SheetMode.OVERLAY
  clickItem: Array<ClickItem> = [
    {
      name: 'push pageOne',
      callback: () => {
        this.pathStack.pushPath({name: 'pageOne'})
      },
      id: 'allSheetPushId'
    }, {
      name: '开启全模态',
      callback: () => {
        this.isShow = true;
      },
      id: 'openAllSheetId'
  }
  ]

  @Builder
  ModalBuilder() {
    ModalComponent()
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

      CommonNavigation({pathStack: this.pathStack}).bindContentCover(this.isShow, this.ModalBuilder)

    }
  }
}