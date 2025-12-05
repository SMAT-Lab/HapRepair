/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@Entry
@Component
struct AExample {
  @State scale2: number = 1
  @State opacity2: number = 1

  build() {
    Column() {
      Navigator({ target: 'pages/Navigator/page_transition', type: NavigationType.Push }) {
        Image($r('app.media.icon')).width('100%').height('100%')
      }
    }.height('100%').width('100%').scale({ x: this.scale2 }).opacity(this.opacity2)
  }

  // 自定义方式1：完全自定义转场过程的效果
  pageTransition() {
    PageTransitionEnter({ duration: 1200, curve: Curve.Linear })
      .onEnter((type: RouteType, progress: number) => {
        this.scale2 = 1
        this.opacity2 = progress
      }) // 进场过程中会逐帧触发onEnter回调，入参为动效的归一化进度(0% -- 100%)
    PageTransitionExit({ duration: 1500, curve: Curve.Ease })
      .onExit((type: RouteType, progress: number) => {
        this.scale2 = 1 - progress
        this.opacity2 = 1
      }) // 退场过程中会逐帧触发onExit回调，入参为动效的归一化进度(0% -- 100%)
  }
}