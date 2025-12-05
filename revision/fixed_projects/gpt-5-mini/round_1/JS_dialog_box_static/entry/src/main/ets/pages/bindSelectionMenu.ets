/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

import web_webview from '@ohos.web.webview';
import Utils from '../../src/test/Util.test';
import { BusinessError, RecordData } from '@ohos.base';
import { Driver, ON } from '@ohos.UiTest';
import emitter from '@ohos.events.emitter';
import {Entry, Component, Column, Row, Button, ClickEvent, Web, Menu, MenuType, Image, WebContextMenuResult, Resource, Builder, ImageFit, MenuItem, Color, WebElementType, WebResponseType, OnContextMenuShowEvent } from "@ohos.arkui.component"
import { State } from "@ohos.arkui.stateManagement"
import { UIContext, Router } from '@ohos.arkui.UIContext'
import { Callback } from '@ohos.base'

interface PreviewBuilderParam {
  previewImage: Resource | string | undefined;
  width: number;
  height: number;
}

class StringFields {
  str: string = '';
}

function px2vp(px: number): number {
  return px / 2;
}

@Builder function PreviewBuilderGlobal($$: PreviewBuilderParam) {
  Column() {
      Image($$.previewImage as Resource | string)
        .objectFit(ImageFit.Fill)
        .autoResize(true)
  }.width($$.width).height($$.height)
}

@Entry
@Component
struct WebComponent {
  controller: web_webview.WebviewController = new web_webview.WebviewController(undefined);
  controller1: web_webview.WebviewController = new web_webview.WebviewController(undefined);

  private result: WebContextMenuResult | undefined = undefined;
  @State previewImage: Resource | string | undefined = undefined;
  @State previewWidth: number = 0;
  @State previewHeight: number = 0;
  @State str: string = '';
  @State userAgentPC:StringFields = new StringFields ();

  @Builder
  MenuBuilder() {
    Menu() {
      MenuItem()
        .onClick((e: ClickEvent) => {
          this.result?.copy();
          this.result?.closeContextMenu();
        })

      MenuItem()
        .onClick((e: ClickEvent) => {
          this.result?.selectAll();
          this.result?.closeContextMenu();
        })
    }
  }

  aboutToAppear() {
    console.info('Entry aboutToAppear');
    let valueChangeEvent: emitter.InnerEvent = {
      eventId: 10,
      priority: emitter.EventPriority.LOW
    }
    emitter.on(valueChangeEvent, this.valueChangeCallBack)
    console.info('Finish aboutToAppear ' + this.valueChangeCallBack);
  }

  private valueChangeCallBack: Callback<emitter.EventData> = (eventData: emitter.EventData): void => {
    console.info('web page valueChangeCallBack');
    if (eventData != null) {
      console.info('valueChangeCallBack:' + JSON.stringify(eventData));
      const data = eventData.data as Record<string, RecordData>;
      if (data && data['ACTION'] != null) {
        console.info('Received ACTION value: ' + data['ACTION']);
        let strValue: string = String(data['ACTION']);
        console.info('strValue has been updated to: ' + strValue);
        this.str = strValue;
        console.info('this.str has been updated to: ' + this.str);
      }
    }
  }

  build() {
    Column() {
      Row() {
        Button("web test").key('webcomponentSelectionMenu').onClick( (e: ClickEvent) => {
          console.info("key==>" + this.str)
          switch (this.str) {
            case "bindSelectionMenuType01": {
              this.userAgentPC.str = this.controller.getUserAgent();
              if ((this.userAgentPC.str).includes("PC")) {
                Utils.emitEvent(true, 8301)
                break;
              }
              break;
            }

            case "bindSelectionMenuType02": {
                Utils.emitEvent(MenuType.PREVIEW_MENU, 8302)
                break;
            }

            case 'SwitchNextPageStatic' : {
              let uiContext: UIContext = this.getUIContext();
              let router: Router = uiContext.getRouter();
              router.pushUrl({ url: 'pages/WebAttributeTest' })
              console.log('push URL success')
            }
          }
        })
      }

      Web({ src: "resource://rawfile/bindSelectionMenu.html", controller: this.controller }).key('bindSelectionMenuType')
        .height(150).width(150).border({ width: 2, color: Color.Red })
        .bindSelectionMenu(WebElementType.IMAGE, this.MenuBuilder, WebResponseType.LONG_PRESS,
          {
            onAppear: () => {},
            onDisappear: () => {
              this.result?.closeContextMenu();
            },
            preview:() => PreviewBuilderGlobal({
              previewImage: this.previewImage,
              width: this.previewWidth,
              height: this.previewHeight
            }),
            menuType: MenuType.PREVIEW_MENU
          })
        .onContextMenuShow((event : OnContextMenuShowEvent): boolean => {
          if (event) {
            this.result = event.result;
            if (event?.param.getLinkUrl()) {
              return false;
            }
            const rawW = event.param.getPreviewWidth?.() ?? 0;
            const rawH = event.param.getPreviewHeight?.() ?? 0;
            this.previewWidth  = px2vp(rawW);
            this.previewHeight = px2vp(rawH);
            Utils.emitEvent(this.previewWidth > 0 && this.previewHeight > 0, 8301)
            if (event.param.getSourceUrl().indexOf("resource://rawfile/") == 0) {
              this.previewImage = event.param.getSourceUrl?.() ?? '';
            } else {
              this.previewImage = event?.param.getSourceUrl();
            }
            return true;
          }
          return false;
        }).margin(80)
    }
  }
}