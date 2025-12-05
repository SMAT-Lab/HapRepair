
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
import { SymbolGlyphModifier } from '@kit.ArkUI';
import { image } from '@kit.ImageKit';

@Component
struct HideToolBarNav {
  @State show: boolean = false;
  // 直接使用一般变量即可
  backButtonIcon: string = 'resources/base/media/icon.png'

  @Builder
  NavigationMenus() {
    Row() {
      Image('resources/base/media/icon.png')
        .width(24)
        .height(24)
      Image('resources/base/media/icon.png')
        .width(24)
        .height(24)
        .margin({ left: 24 })
    }
  }

  build() {
    NavDestination() {
      Stack({ alignContent: Alignment.Center }) {
        Button('switch')
          .onClick(() => {
            this.show = !this.show;
          })
          .height('10%')
          .id('switchId')
      }.width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
    .title('Destination Title')
    .toolbarConfiguration([
      { value: 'A' },
      { value: 'B' },
      { value: 'C' },
      { value: 'D' },
      { value: 'E' },
      { value: 'F' }
    ])
    .hideToolBar(this.show, this.show)
    .backButtonIcon(this.backButtonIcon)
    .menus(this.NavigationMenus)
  }
}

@Component
struct SymbolGlyphModifierBackBtn {
  build() {
    NavDestination() {
      Stack({ alignContent: Alignment.Center }) {
      }.width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
    .title('Destination Title')
    .backButtonIcon(new SymbolGlyphModifier($r('sys.symbol.ohos_folder_badge_plus')));
  }
}

@Component
struct PixelMapBackBtn {
  @State imagePixelMap: image.PixelMap | undefined = undefined;

  async aboutToAppear() {
    this.imagePixelMap = await this.getPixmapFromMedia($r('app.media.icon'))
  }

  async getPixmapFromMedia(resource: Resource) {
    let unit8Array = await getContext(this)?.resourceManager?.getMediaContent({
      bundleName: resource.bundleName,
      moduleName: resource.moduleName,
      id: resource.id
    })
    let imageSource = image.createImageSource(unit8Array.buffer.slice(0, unit8Array.buffer.byteLength))
    let createPixelMap: image.PixelMap = await imageSource.createPixelMap({
      desiredPixelFormat: image.PixelMapFormat.RGBA_8888
    })
    await imageSource.release()
    return createPixelMap
  }

  build() {
    NavDestination() {
      Stack({ alignContent: Alignment.Center }) {
      }.width('100%')
      .height('100%')
    }
    .width('100%')
    .height('100%')
    .title('Destination Title')
    .backButtonIcon( this.imagePixelMap);
  }
}

@Entry
@Component
struct HideToolBarNavPage {
  @Provide('navPathStack') navPathStack: NavPathStack = new NavPathStack();

  @Builder
  PageBuilder(name: string, param?: Object) {
    if (name === 'hideToolBarNav') {
      HideToolBarNav();
    } else if (name === 'symbolGlyphModifierBack') {
      SymbolGlyphModifierBackBtn();
    } else if (name === 'pushPixelMap') {
      PixelMapBackBtn();
    }

  }

  build() {
    Navigation(this.navPathStack) {
      Stack({ alignContent: Alignment.Center }) {
        Column() {
          Button('push')
            .onClick(() => {
              this.navPathStack.pushPathByName('hideToolBarNav', null);
            })
            .id('pushId')
            .height('10%')
          Button('pushSymbol')
            .onClick(() => {
              this.navPathStack.pushPathByName('symbolGlyphModifierBack', null);
            })
            .height('10%')
          Button('pushPixelMap')
            .onClick(() => {
              this.navPathStack.pushPathByName('pushPixelMap', null);
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
    .height('90%')
    .mode(NavigationMode.Stack)
    .navDestination(this.PageBuilder)
    .titleMode(NavigationTitleMode.Mini)
    .toolbarConfiguration([
      { value: 'A' },
      { value: 'B' },
      { value: 'C' },
      { value: 'D' },
      { value: 'E' },
      { value: 'F' }
    ])
    .hideToolBar(true, false)
  }
}