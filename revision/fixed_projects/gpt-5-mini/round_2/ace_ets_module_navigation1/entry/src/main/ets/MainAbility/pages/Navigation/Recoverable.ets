/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@Entry
@Component
struct RecoverableDemo {
  @State navigationOpenReceover:boolean = true
  @State navigationrecstring:string = '可恢复'
  @Provide('pageInfos')pageInfos: NavPathStack = new NavPathStack()
  needRecovery: boolean = false;

  @Builder
  pageMap(name:string){
  }


  build() {
    Column(){
      Navigation(this.pageInfos){
        Scroll(){
          Column(){
            Text('这是NavIndex 首页').fontStyle(FontStyle.Italic).fontWeight(FontWeight.Bolder).padding({top:10,bottom:10})
            Button('navigation：'+this.navigationrecstring).onClick(()=>{
              if(this.navigationOpenReceover===true){
                this.navigationOpenReceover=false
                this.navigationrecstring='当前页面不可恢复'
              }else{
                this.navigationOpenReceover=true
                this.navigationrecstring='当前页面可恢复'
              }
            }).margin(20)
            Button('push pageOne', { stateEffect: true, type: ButtonType.Capsule })
              .width('80%')
              .height(40)
              .margin({ bottom: 10 })
              .onClick(()=>{
                this.pageInfos.pushPath({ name: 'pageOne' ,})
              })
            Button('routerPage1_recoverable=true', { stateEffect: true, type: ButtonType.Capsule })
              .width('80%')
              .height(40)
              .margin({ bottom: 10 })
          }.backgroundColor('#FFFF77')
          .height('100%')
        }
      }.title('NavIndex')
      .recoverable(this.navigationOpenReceover)
      .id('testNavigation')
      .navDestination(this.pageMap)
      .height('50%')

      Button('update recoverable')
        .onClick(() => {
          this.navigationOpenReceover = !this.navigationOpenReceover;
        })
        .id('update')

    }
  }
}