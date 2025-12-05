
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

import {
  Button,
  ClickEvent,
  Column,
  Component,
  Entry,
  KeyEvent,
  OnAlertEvent,
  OnConfirmEvent,
  OnPageBeginEvent,
  OnPageEndEvent,
  OnProgressChangeEvent,
  OnPromptEvent,
  OnScaleChangeEvent,
  OnTitleReceiveEvent,
  Row,
  Web
} from '@ohos.arkui.component';
import { State } from '@ohos.arkui.stateManagement';
import webview from '@ohos.web.webview';
import { BusinessError, Callback, RecordData } from '@ohos.base';
import emitter from '@ohos.events.emitter';
import Utils from '../../src/test/Util.test';
import { Driver, PointerMatrix } from '@ohos.UiTest';
import cert from '@ohos.security.cert';
import { UIContext, Router } from '@ohos.arkui.UIContext'

@Entry
@Component
struct WebAttributeTest {
  controller: webview.WebviewController = new webview.WebviewController(undefined);
  @State str: string = ""
  initialScale: number = 100
  newScale: number = 0
  oldScale: number = 0
  geolocationAccess: boolean = true
  blockNetwork: boolean = false
  progress: string = ""
  geoShow: boolean = false
  @State pageBegin: string = ""
  @State pageEnd: string = ""
  @State alert: string = ""
  @State confirm: string = ""
  @State prompt: string = ""
  errorReceive: string = ""
  isHttpErrorReceive: boolean = false
  isLargeThan: boolean = false
  httpErrorReceive: number = 0
  isDownloadStart: boolean = false
  newUrl: string = ""
  isRefreshed: boolean = false
  loadedUrl: string = ""
  webInit: boolean = false
  webCreated: boolean = false
  searchResultCount: number = -1
  onInterceptKeyEventCode: number = 0;
  onFaviconReceivedCalled: boolean = false
  outputStr: string = ''
  handleCancel: boolean = false;
  title: string = ''
  onRequestSelectedCalled: boolean = false
  testConsole: boolean = false

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
    Row() {
      Column() {
        Button("web click")
          .key('WebAttributeTestButton')
          .focusable(true)
          .onFocus(() => {
            console.log("onFocus==>")
          })
          .onBlur(() => {
            console.log("onBlur==>")
          })
          .onClick((e: ClickEvent) => {
            console.info("key==>" + this.str)
            switch (this.str) {
              case "emitOnAlert001": {
                try {
                  this.controller.loadUrl("resource://rawfile/alert.html");
                  this.controller.runJavaScript('alertTest()')
                  setTimeout(() => {
                    Utils.emitEvent(this.alert, 70017)
                  }, 500);
                } catch (error: BusinessError) {
                  console.error(`ErrorCode: ${error.code},  Message: ${error.message}`);
                }
                break;
              }
              case "emitOnConfirm001": {
                try {
                  this.controller.loadUrl("resource://rawfile/confirm.html");
                  setTimeout(() => {
                    this.controller.runJavaScript('confirmTest()')
                  }, 800);
                  setTimeout(() => {
                    Utils.emitEvent(this.confirm, 70018)
                  }, 1000);
                } catch (error: BusinessError) {
                  console.error(`ErrorCode: ${error.code},  Message: ${error.message}`);
                }
                break;
              }
              case "emitOnPrompt001": {
                try {
                  this.controller.loadUrl("resource://rawfile/prompt.html");
                  setTimeout(() => {
                    this.controller.runJavaScript("promptTest()")
                  }, 800);
                  setTimeout(() => {
                    Utils.emitEvent(this.prompt, 70019)
                  }, 1000);
                } catch (error: BusinessError) {
                  console.error(`ErrorCode: ${error.code},  Message: ${error.message}`);
                }
                break;
              }
              case 'SwitchNextPageStatic' : {
                let uiContext: UIContext = this.getUIContext();
                let router: Router = uiContext.getRouter();
                router.pushUrl({ url: 'pages/WebViewEnumAndFontLibrary' })
                console.log('push URL success')
              }
            }
          })

        Web({ src: 'resource://rawfile/index2.html', controller: this.controller })
          .key('webView')
          .height('50%')
          .fileAccess(true)
          .width('100%')
          .javaScriptAccess(true)
          .horizontalScrollBarAccess(true)
          .verticalScrollBarAccess(true)
          .initialScale(this.initialScale)
          .geolocationAccess(this.geolocationAccess)
          .databaseAccess(true)
          .onPageBegin((event: OnPageBeginEvent): void => {
            if (event) {
              this.pageBegin = event.url
              console.log("onPageBegin==> event.url:" + event.url)
            }
          })
          .onPageEnd((event: OnPageEndEvent): void => {
            if (event) {
              this.pageEnd = event.url
              console.log("onPageEnd==> event.url:" + event.url)
              console.log("onPageEnd==> this.pageEnd:" + this.pageEnd)
            }
          })
          .onAlert((event: OnAlertEvent): boolean => {
            if (event) {
              console.log("onAlert==>event.url:" + event.url)
              console.log("onAlert==>event.message:" + event.message)
              this.alert = event.message
            }
            return false
          })
          .onConfirm((event: OnConfirmEvent): boolean => {
            if (event) {
              console.log("onConfirm==>event.url:" + event.url)
              console.log("onConfirm==>event.message:" + event.message)
              this.confirm = event.message;
              if (this.handleCancel) {
                event.result.handleCancel()
              } else {
                event.result.handleConfirm()
              }
            }
            return true
          })
          .onPrompt((event: OnPromptEvent): boolean => {
            if (event) {
              console.log("onPrompt==>event.url:" + event.url)
              console.log("onPrompt==>event.message:" + event.message)
              console.log("onPrompt==>event.value:" + event.value)
              this.prompt = event.message;
              event.result.handlePromptConfirm('handlePromptConfirm')
            }
            return true
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}