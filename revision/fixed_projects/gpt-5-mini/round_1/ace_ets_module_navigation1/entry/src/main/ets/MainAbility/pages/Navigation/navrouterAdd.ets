
/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@Entry
@Component
struct NavRouterExample {
  private arr: number[] = [0, 1, 2, 3]
  isActive: boolean = false;
  @State isIndex: number = 0

  build() {
    Column() {
      Navigation() {
        List({ space: 12, initialIndex: 0 }) {
          ForEach(this.arr, (item: number, index: number) => {
            ListItem() {
              NavRouter({name:'',param:''}) {
                Row() {
                  Image($r('app.media.icon')).width(30).height(30).borderRadius(30).margin({ left: 3, right: 10 })
                  Text(`NavRouter${item + 1}`)
                    .fontSize(22)
                    .fontWeight(500)
                    .textAlign(TextAlign.Center)
                }
                .width(180)
                .height(72)
                .backgroundColor(this.isIndex === index ? '#ccc' : '#fff')
                .borderRadius(24)

                NavDestination() {
                  Text(`我是NavDestination第${item + 1}页内容`).fontSize(50)
                  Flex({ direction: FlexDirection.Row }) {
                    Row() {
                      Image($r('app.media.icon')).width(40).height(40).borderRadius(40).margin({ right: 15 })
                      Text('今天共有七节课').fontSize(30)
                    }.padding({ left: 15 })
                  }
                }.backgroundColor('#ccc')
                .title(`NavDestination${item + 1}`)
              }.onStateChange((isActivated: boolean) => {
                this.isIndex = index
                console.info('the mode is ' + NavRouteMode.PUSH_WITH_RECREATE);
                console.info('the mode is ' + NavRouteMode.REPLACE);
              }).mode(NavRouteMode.PUSH)
            }
          }, (item: string) => item)
        }
        .height('100%')
        .margin({ top: 12, left: 12 })
      }
      .mode(NavigationMode.Split)
      .hideTitleBar(true)
      .key('addOne')
      .hideToolBar(true)
    }.height('100%')
  }
}