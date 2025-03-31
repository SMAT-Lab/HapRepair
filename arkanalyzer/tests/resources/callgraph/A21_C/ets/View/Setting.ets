
import promptAction from '@ohos.promptAction'
import router from '@ohos.router'
import ItemData from '../MyClass/ItemData'
import mainViewModel from '../MyClass/MainViewModel'
@Component
export struct Setting {
  @Builder settingCell(item: ItemData) {
    Row() {
      Row({ space: 12 }) {
        Image(item.img)
          .width(24)
          .height(24)
        Text(item.title)
      }
      if(item.others === null){
        Image($r("app.media.right"))
          .width(24)
          .height(24)
      }else {
        Toggle({type: ToggleType.Switch, isOn: false})
      }
    }
    .justifyContent(FlexAlign.SpaceBetween)
    .width('100%')
    .padding(10)

  }
  @State message: string = 'Hello World'
  build() {
    Row() {
      Column() {
        Column(){
          Row(){
            Image($r('app.media.img_2'))
              .width(60)
              .height(60)
            Column(){
              Text('User001')
                .fontSize(25)
                .fontWeight(FontWeight.Bold)
              Text('111@qq.com')
                .fontSize(14)
                .margin({
                  top: 5
                })
                .fontColor('#a0a0a0')
            }
            .justifyContent(FlexAlign.SpaceBetween)
            .alignItems(HorizontalAlign.Start)
            .padding(12)
          }
          .padding(20)
          .borderRadius(10)
          .margin(15)
          .backgroundColor('#fff')
          .width("92%")
          List(){
            ForEach(mainViewModel.getSettingListData(),
              (item: ItemData) => {
                ListItem() {
                  this.settingCell(item)
                }
              }, item => JSON.stringify(item))
          }
          .divider({color: '#efefef', strokeWidth: 1})
          .width("95%")
          .padding(5)
          .borderRadius(5)
          .backgroundColor('#fff')
        }

        Button('退出登录')
          .width('90%')
          .fontColor('#ff0000')
          .backgroundColor('#e0e0e0')
          .onClick(()=>{
            promptAction.showToast({
              message: '退出成功',
              duration: 2000
            })
            setTimeout(() => {
              router.replaceUrl({
                url: "pages/Index"
              })
            }, 2000)
          })
      }
      .justifyContent(FlexAlign.SpaceBetween)
      .width('100%')
      .height('95%')
    }
    .alignItems(VerticalAlign.Top)
    .height('100%')
    .backgroundColor('#efefef')
  }
}