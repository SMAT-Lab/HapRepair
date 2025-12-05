
/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
struct bindSelectionMenuTest {
  controller: TextController = new TextController();
  options: TextOptions = { controller: this.controller };
  @State onMenuShow: string = '';
  @State onMenuHide: string = '';

  build() {
    Column() {
      Text(undefined, this.options) {
        Span('HelloWorld')
      }.fontSize('15vp')
      .id('text')
      .copyOption(CopyOptions.InApp)
      .bindSelectionMenu(TextSpanType.DEFAULT, this.panel, TextResponseType.DEFAULT, {
        onDisappear: () => {
          console.info(`自定义选择菜单关闭时回调`);
        },
        onAppear: () => {
          console.info(`自定义选择菜单弹出时回调`);
        },
        onMenuShow: () => {
          console.log('wgl onMenuShow succ');
          this.onMenuShow = 'succ';
        },
  handleTextSelectionChange() {
    // 处理文本选中区域变化的逻辑
  }
        onMenuHide: () => {
          console.log('wgl onMenuHide succ');
          this.onMenuHide = 'succ';
        }
      })
      .onTextSelectionChange((selectionStart: number, selectionEnd: number) => {
        this.handleTextSelectionChange()
      })
      .height('15%')
      .textAlign(TextAlign.Center)
      Text(undefined, this.options) {
        ImageSpan($r('app.media.icon'))
          .width('50vp')
          .height('50vp')
          .objectFit(ImageFit.Fill)
          .verticalAlign(ImageSpanAlignment.CENTER)
      }
      .id('image')
      .textAlign(TextAlign.Center)
      .copyOption(CopyOptions.InApp)
      .bindSelectionMenu(TextSpanType.DEFAULT, this.panel, TextResponseType.DEFAULT, {
        onDisappear: () => {
          console.info(`自定义选择菜单关闭时回调`);
        },
        onAppear: () => {
          console.info(`自定义选择菜单弹出时回调`);
        },
        onMenuShow: () => {
          console.log(' onMenuShow succ');
          this.onMenuShow = 'succ';
        },
        onMenuHide: () => {
          console.log(' onMenuHide succ');
          this.onMenuHide = 'succ';
        }
      })
      .onTextSelectionChange((selectionStart: number, selectionEnd: number) => {
        console.info(`文本选中区域变化回调, selectionStart: ${selectionStart}, selectionEnd: ${selectionEnd}`);
      })
      .height('15%')
      Text(undefined, this.options) {
        Span('HelloWorld')
        ImageSpan($r('app.media.icon'))
          .width('50vp')
          .height('50vp')
          .objectFit(ImageFit.Fill)
          .verticalAlign(ImageSpanAlignment.CENTER)
      }.fontSize('15vp')
      .id('mixed')
      .copyOption(CopyOptions.InApp)
      .bindSelectionMenu(TextSpanType.DEFAULT, this.panel, TextResponseType.DEFAULT, {
        onDisappear: () => {
          console.info(`自定义选择菜单关闭时回调`);
        },
        onAppear: () => {
          console.info(`自定义选择菜单弹出时回调`);
        },
        onMenuShow: () => {
          console.log('onMenuShow succ');
          this.onMenuShow = 'succ';
        },
  handleTextSelectionChange(selectionStart: number, selectionEnd: number) {
    // 处理文本选中区域变化的逻辑
  }
        onMenuHide: () => {
          console.log('onMenuHide succ');
          this.onMenuHide = 'succ';
        }
      })
      .onTextSelectionChange((selectionStart: number, selectionEnd: number) => {
        this.handleTextSelectionChange(selectionStart, selectionEnd)
      })
      .height('15%')
      .textAlign(TextAlign.Center)
    }
    .width('100%')
    .height('100%')
  }
  @Builder
  panel() {
    Column() {
      Text('自定义菜单')
    }.width(130).height(20).backgroundColor(Color.Red)
  }
}