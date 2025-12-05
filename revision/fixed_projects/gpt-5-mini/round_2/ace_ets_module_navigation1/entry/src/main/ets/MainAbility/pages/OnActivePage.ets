@Entry
@Component
struct OnActivePage {
  build() {
    Flex({ direction:FlexDirection.Column, alignItems:ItemAlign.Center, justifyContent: FlexAlign.Center }){
       Button('Overlay').onClick(() => {
           this.getUIContext().getRouter().pushUrl({url: 'MainAbility/pages/OverlayTest'})
       }).id('overLayBtnId').height('10%')
      Button('基础场景').onClick(() => {
        this.getUIContext().getRouter().pushUrl({url: 'MainAbility/pages/Basic'})
      }).id('baseBtnId').height('10%')
      Button('CustomDialog').onClick(() => {
        this.getUIContext().getRouter().pushUrl({url: 'MainAbility/pages/CustomDialogTest'})
      }).id('dialogBtnId').height('10%')
      Button('半模态场景').onClick(() => {
        this.getUIContext().getRouter().pushUrl({url: 'MainAbility/pages/BasicSheet01'})
      }).id('halfBtnId').height('10%')
      Button('全模态场景').onClick(() => {
        this.getUIContext().getRouter().pushUrl({url: 'MainAbility/pages/BasicSheet02'})
      }).id('allBtnId').height('10%')
    }
  }
}