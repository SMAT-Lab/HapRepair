
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
import Utils from '../../src/test/Util.test';
import webView from '@ohos.web.webview';
import emitter from '@ohos.events.emitter';
import { Driver, ON } from '@ohos.UiTest';
import { BusinessError, RecordData } from '@ohos.base';
import {
  OnRenderExitedEvent,
  OnPageBeginEvent,
  RenderProcessNotRespondingReason,
  RenderProcessNotRespondingData,
  Entry,
  Component,
  Column,
  Row,
  Button,
  ClickEvent,
  Web,
  LargestContentfulPaint,
  Flex,
  Image,
  Builder,
  Text,
  FlexDirection,
  FlexAlign,
  ItemAlign,
  ForEach,
  Alignment,
  Divider,
  $r,
} from '@ohos.arkui.component'
import { State } from '@ohos.arkui.stateManagement'
import { Callback } from '@ohos.base'


class StringFields {
  str: string = '';
}



@Entry
@Component
struct WebViewEnumAndFontLibrary {
  stringObj: StringFields  = new StringFields ();


  controller: webView.WebviewController = new webView.WebviewController(undefined);
  @State access: boolean = true
  @State minFontSize: int = 1
  @State webResultText: string = ''
  @State minLogicalFontSize: int = 13

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
        console.info('Received ACTION vaclue: ' + data['ACTION']);
        let strValue: string = String(data['ACTION']);
        console.info('strValue has been updated to: ' + strValue);
        this.stringObj.str = strValue;
        console.info('this.str has been updated to: ' + this.stringObj.str);
      }
    }
  }

  build() {
    Column() {
      Row() {
        Button('web click')
          .key('buttonEnumAndFontLibrary')
          .onClick((e: ClickEvent) => {
          console.info('key==>' + this.stringObj.str )
          switch (this.stringObj.str ) {
            case 'emitFileSelectorModeFileOpenMode': {
                Utils.emitEvent(0, 861)
            }
          }
        })
      }


        Text(this.webResultText).fontSize(20)

      Web({src: 'resource://rawfile/indexJru.html', controller: this.controller })
        .overviewModeAccess(true)
        .key('webCenter')
        .minFontSize(this.minFontSize)
        .minLogicalFontSize(this.minLogicalFontSize)
        .forceDarkAccess(this.access)
        .fileAccess(true)
    }
  }
}