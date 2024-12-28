import router from '@ohos.router';
import promptAction from '@ohos.promptAction';
import {Home} from '../View/Home'
import {Setting} from '../View/Setting'
@Entry
@Component
struct LoginOK {
  @State currentIndex: number = 0;
  private tabsController: TabsController = new TabsController();
  //private fruits: string[]=['5s','10s','30s','50s'];
  @Builder TabBuilder(title: string, index: number, selectedImg: Resource, normalImg: Resource) {
    Column(){
      Image(this.currentIndex === index ? selectedImg : normalImg)
        .width(30)
        .height(30)
      Text(title)
        .fontColor(this.currentIndex === index ? '#077afb': '#d4d2d6')
        .fontSize(12)
    }
    .onClick(() => {
      this.currentIndex = index;
      this.tabsController.changeIndex(index);
    })
  }
   build() {
    Column(){

      Tabs({barPosition: BarPosition.End, controller: this.tabsController}) {
        TabContent(){
          Home()
        }
        .tabBar(this.TabBuilder('首页',0,$r('app.media.home_HL'), $r('app.media.home')))

        TabContent(){
          Setting()
        }
        .tabBar(this.TabBuilder('我的', 1, $r('app.media.user_HL'), $r('app.media.User')))
      }
      .barWidth('100%')
      .barHeight(56)
      .barMode(BarMode.Fixed)
      .onChange((index: number) => {
        this.currentIndex = index
      })
    }

    }

}