
import { CommonNavigation } from './CommonNavigation';

export interface ClickItem {
  name: string;
  callback: () => void;
  id?: string;
}

@CustomDialog
struct DialogComponent {
  controller: CustomDialogController
  cancel: () => void = () => {}
  pathStack: NavPathStack = new NavPathStack()

  @Builder
  pageBuilder(name: string) {
    if (name == "pageOne") {
      NavDestination() {

      }.title('PageOne')
      .onActive((reason)=> {
        console.log('trigger CustomDialog pageOne onActive: ' + reason);
      })
      .onInactive((reason) => {
        console.log('trigger CustomDialog pageOne onInactive: ' + reason);
      })
    }
  }
  build() {
    Column() {
      Button('关闭弹窗CustomDialog').width(300).height(30).onClick(() => {
        this.controller.close();
      })
      .id('closeDialogId')
      Navigation(this.pathStack){}.navDestination(this.pageBuilder).title('CustomDialog')
    }
  }
}

@Entry
@Component
struct CustomDialogTest {
  pathStack: NavPathStack = new NavPathStack();
  @State isShowInSubWindow: boolean = false;
  isModal: boolean = false;
  dialogController: CustomDialogController = new CustomDialogController({
    builder: DialogComponent(),
    showInSubWindow: this.isShowInSubWindow
  });

  clickItem: Array<ClickItem> = [
   {
    name: 'push pageOne',
    callback: () => {
      this.pathStack.pushPath({name: 'pageOne'})
    },
     id: 'dialogPushId'
  },
    {
      name: '打开CustomDialog',
      callback: () => {
        this.isShowInSubWindow = false;
        this.isModal = true;
        this.dialogController.open();
      },
      id: 'openDialogId'
    }
  ]
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
    }
  }
}