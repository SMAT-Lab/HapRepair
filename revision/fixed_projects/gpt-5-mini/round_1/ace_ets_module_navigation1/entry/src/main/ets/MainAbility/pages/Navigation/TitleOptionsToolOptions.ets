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

let COLOR1: string = '#80004AAF';
let COLOR2: string = '#802787D9';
let BLUR_STYLE_1: BlurStyle = BlurStyle.BACKGROUND_THIN;
let BLUR_STYLE_2: BlurStyle = BlurStyle.BACKGROUND_THICK;

@Component
struct BackComponent {
  build() {
    Row() {
      Column() {
      }
      .height('100%')
      .backgroundColor('#3D9DB4')
      .layoutWeight(9)

      Column() {
      }
      .height('100%')
      .backgroundColor('#17A98D')
      .layoutWeight(9)

      Column() {
      }
      .height('100%')
      .backgroundColor('#FFC000')
      .layoutWeight(9)
    }
    .height('100%')
    .width('100%')
  }
}

@Component
struct ColorAndBlur {
  @State useColor1: boolean = true;
  @State useBlur1: boolean = true;

  build() {
    NavDestination() {
      Stack({ alignContent: Alignment.Center }) {
        BackComponent()
          .width('100%')
          .height('100%')
        Column() {
          Button('switch')
            .onClick(() => {
              this.useColor1 = !this.useColor1;
              this.useBlur1 = !this.useBlur1;
            })
            .height('10%')
            .margin({top: '30%'})
        }
        .width('100%')
        .height('100%')
      }.width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
    .title('Destination Title', {
      backgroundColor: this.useColor1 ? COLOR1 : COLOR2,
      backgroundBlurStyle: this.useBlur1 ? BLUR_STYLE_1 : BLUR_STYLE_2,
      barStyle: BarStyle.STACK
    })
    .menus([
      { value: 'A' },
      { value: 'B' },
      { value: 'C' },
      { value: 'D' },
    ])
    .toolbarConfiguration([
      { value: 'A' },
      { value: 'B' },
      { value: 'C' },
      { value: 'D' },
      { value: 'E' },
      { value: 'F' }
    ])
  }
}

@Entry
@Component
struct TitleOptionsToolOptions {
  @Provide('navPathStack') navPathStack: NavPathStack = new NavPathStack();
  @State useColor1: boolean = true;
  @State useBlur1: boolean = true;

  @Builder
  PageBuilder(name: string, param?: Object) {
    if (name === 'NavigationMenu') {
      ColorAndBlur();
    }
  }

  build() {
    Navigation(this.navPathStack) {
      Stack({ alignContent: Alignment.Center }) {
        BackComponent()
          .width('100%')
          .height('100%')
        Column() {
          Button('switch')
            .onClick(() => {
              this.useColor1 = !this.useColor1;
              this.useBlur1 = !this.useBlur1;
            })
            .height('10%')
            .margin({top: '20%'})
          Button('push')
            .onClick(() => {
              this.navPathStack.pushPathByName('NavigationMenu', null);
            })
            .height('10%')
        }
        .width('100%')
        .height('80%')
      }
      .width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
    .mode(NavigationMode.Stack)
    .navDestination(this.PageBuilder)
    .title('NavTitle', {
      backgroundColor: this.useColor1 ? COLOR1 : COLOR2,
      backgroundBlurStyle: this.useBlur1 ? BLUR_STYLE_1 : BLUR_STYLE_2,
      barStyle: BarStyle.STACK,
    })
    .titleMode(NavigationTitleMode.Mini)
    .menus([
      { value: 'A' },
      { value: 'B' },
      { value: 'C' },
      { value: 'D' },
    ]
    )
    .toolbarConfiguration([
      { value: 'A' },
      { value: 'B' },
      { value: 'C' },
      { value: 'D' },
      { value: 'E' },
      { value: 'F' }
    ], {
      backgroundColor: this.useColor1 ? COLOR1 : COLOR2,
      backgroundBlurStyle: this.useBlur1 ? BLUR_STYLE_1 : BLUR_STYLE_2,
      barStyle: BarStyle.STACK,
    })
  }
}