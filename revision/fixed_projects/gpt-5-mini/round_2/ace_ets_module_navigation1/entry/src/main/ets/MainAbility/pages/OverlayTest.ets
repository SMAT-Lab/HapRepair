import { BusinessError } from '@kit.BasicServicesKit';
import { ComponentContent } from '@kit.ArkUI';
import { CommonNavigation } from './CommonNavigation';


@Component
struct OverlayComponent {
  build() {
    Column() {
      Button('关闭Overlay').onClick(() => {
        this.getUIContext().getOverlayManager().hideAllComponentContents()
      }).id('closeOverLayId')
    }.backgroundColor(Color.White)
  }
}

@Builder
function overlayBuilder() {
  OverlayComponent()
}

@Entry
@Component
struct OverlayTest {
  pathStack: NavPathStack = new NavPathStack();
  builderComponent: ComponentContent<never> = new ComponentContent(this.getUIContext(), wrapBuilder(overlayBuilder))
  clickItems: Array<ClickItem> = [{
    name: 'push pageOne',
    callback: () => {
      this.pathStack.pushDestination({name: 'pageOne', param: 'overlay'}).then(() => {
        console.log('testOnActive pageOne success');
      }).catch((err: BusinessError) => {
        console.log(`testOnActive pageOne failed: ${err.code}, message: ${err.message}`);
      })
    },
    id: 'overLayPushId'
  }, {
    name: '开启Overlay',
    callback: () => {
      this.getUIContext().getOverlayManager().addComponentContent(this.builderComponent)
      this.getUIContext().getOverlayManager().showComponentContent(this.builderComponent)
    },
    id: 'openOverLayId'
  }
  ]

  build() {
    Column() {
      ForEach(this.clickItems, (item: ClickItem) => {
        Button(item.name).width(300).height(30).onClick(item.callback).margin(10).id(item.id)
      })
       CommonNavigation({pathStack: this.pathStack, title: 'Overlay Page Navigation'})
    }
  }
}

export interface ClickItem {
  name: string;
  callback: () => void;
  id?: string;
}