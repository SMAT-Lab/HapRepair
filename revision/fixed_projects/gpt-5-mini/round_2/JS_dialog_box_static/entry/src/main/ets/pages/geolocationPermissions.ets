
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

import webView from '@ohos.web.webview';
import { $rawfile, Entry, Component, Column, Row, Button, ClickEvent, Web } from '@ohos.arkui.component'
import { State } from '@ohos.arkui.stateManagement'
import { BusinessError, AsyncCallback, RecordData } from '@ohos.base';
import Utils from '../../src/test/Util.test';
import emitter from '@ohos.events.emitter';
import { JsProxyObject } from '../../src/test/Interfaces';
import { Callback } from '@ohos.base'
import { UIContext, Router } from '@ohos.arkui.UIContext'

class StringFields {
  str: string = '';
}

@Entry
@Component
struct geolocationPermissions {
  stringObj: StringFields  = new StringFields ();
  controller: webView.WebviewController = new webView.WebviewController(undefined);
  str: string = ''
  javaScriptAccess: boolean = true;
  incognito: boolean = false;
  origin: string = 'resource://rawfile/'
  no_error: string = 'no error happened'


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
        this.stringObj.str = strValue;
        console.info('this.str has been updated to: ' + this.stringObj.str);
      }
    }
  }

  private jsObj: JsProxyObject = {
    test: (res: object) => {
      Utils.emitEvent(String(res), 102);
    },
    showString: (str: string) => {
      console.info('ets toString:' + String(str));
    },
    register: (res: object) => {
      Utils.emitEvent(String(res), 86);
      return 'web222'
    }
  } as JsProxyObject

  build() {
    Column(undefined) {
      Row(undefined) {
        Button('GeolocationPermissions click')
          .key('GeolocationPermissionsButton').onClick((e: ClickEvent) => {
          console.info('key==>' + this.stringObj.str)
          switch (this.stringObj.str) {

            case 'testGetStoredGeolocation001': {
              try {
                webView.GeolocationPermissions.allowGeolocation('file:///')
                webView.GeolocationPermissions.getStoredGeolocation((error, origins) => {
                  if (error?.code) {
                    console.log('getStoredGeolocationAsync error: ' + JSON.stringify(error))
                    return;
                  }
                  if (origins) {
                    let origins_str: string = origins.join()
                    console.log('getStoredGeolocationAsync origins: ' + origins_str);
                    Utils.emitEvent(origins_str, 30060);
                  }
                });
              } catch (error: BusinessError) {
                console.error(`ErrorCode: ${error.code}, Message: ${error.message}`);
              }
              break;
            }


            case 'testGetStoredGeolocation002': {
              try {
                webView.GeolocationPermissions.allowGeolocation('file:///')
                webView.GeolocationPermissions.getStoredGeolocation((error, origins) => {
                  if (error?.code) {
                    console.log('getStoredGeolocationPromise error: ' + JSON.stringify(error))
                    return;
                  }
                  if (origins) {
                    let origins_str: string = origins.join()
                    console.log('getStoredGeolocationAsync origins: ' + origins_str);
                    Utils.emitEvent(origins_str, 30061);
                  }
                });
              } catch (error: BusinessError) {
                console.error(`ErrorCode: ${error.code}, Message: ${error.message}`);
              }
              break;
            }


            case 'testGetStoredGeolocation003': {
              try {
                webView.GeolocationPermissions.deleteGeolocation(this.origin, this.incognito)
                webView.GeolocationPermissions.getStoredGeolocation((error, origins) => {
                  if (error?.code) {
                    console.log('getStoredGeolocationAsync error: ' + JSON.stringify(error))
                    return;
                  }
                  if (origins) {
                    let origins_str: string = origins.join()
                    console.log('getStoredGeolocationAsync origins: ' + origins_str);
                    Utils.emitEvent(origins_str, 100011);
                  }
                });
              } catch (error: BusinessError) {
                console.error(`ErrorCode: ${error.code}, Message: ${error.message}`);
              }
              break;
            }

            case 'testGetStoredGeolocation004': {
              try {
                webView.GeolocationPermissions.deleteGeolocation(this.origin, this.incognito)
                webView.GeolocationPermissions.getStoredGeolocation((error, origins) => {
                  if (error?.code) {
                    console.log('getStoredGeolocationAsync error: ' + JSON.stringify(error))
                    return;
                  }
                  if (origins) {
                    let origins_str: string = origins.join()
                    console.log('getStoredGeolocationAsync origins: ' + origins_str);
                    Utils.emitEvent(origins_str, 100012);
                  }
                });
              } catch (error: BusinessError) {
                console.error(`ErrorCode: ${error.code}, Message: ${error.message}`);
              }
              break;
            }

            case 'testDeleteAllGeolocation001': {
              let text: undefined | String = '123'
              webView.GeolocationPermissions.allowGeolocation('resource://rawfile/')
              webView.GeolocationPermissions.deleteAllGeolocation(this.incognito);
              webView.GeolocationPermissions.getStoredGeolocation((error, origins) => {
                if (error?.code) {
                  console.log('error:' + JSON.stringify(error));
                  text = origins + ',error ,' + JSON.stringify(error);
                  return;
                }
                text = origins?.join();
                Utils.emitEvent(text as String, 30055 as Int)
              })
              break;
            }

            case 'testGetAccessibleGeolocation001': {
              try {
                webView.GeolocationPermissions.allowGeolocation(this.origin)
                webView.GeolocationPermissions.getAccessibleGeolocation(this.origin)
                  .then((result: Boolean) => {
                    console.log('getAccessibleGeolocation result: ' + result);
                    Utils.emitEvent(result, 100008);
                  }).catch((error: Error) => {
                  console.log('getAccessibleGeolocation error: ' + JSON.stringify(error));
                  Utils.emitEvent(error, 100008);
                });
              } catch (error: BusinessError) {
                let e: BusinessError = error as BusinessError;
                console.error(`ErrorCode: ${e.code},  Message: ${e.message}`);
                Utils.emitEvent(e.code, 100008);
              }
              break;
            }

            case 'testGetAccessibleGeolocation002': {
              try {
                webView.GeolocationPermissions.allowGeolocation(this.origin)
                webView.GeolocationPermissions.getAccessibleGeolocation(this.origin)
                  .then((result: Boolean) => {
                    console.log('getAccessibleGeolocationPromise result: ' + result);
                    Utils.emitEvent(result, 100009);
                  }).catch((error: Error) => {
                  console.log('getAccessibleGeolocationPromise error: ' + JSON.stringify(error));
                  Utils.emitEvent(error, 100009);
                });
              } catch (error: BusinessError) {
                let e: BusinessError = error as BusinessError;
                console.error(`ErrorCode: ${e.code},  Message: ${e.message}`);
                Utils.emitEvent(e.code, 100009);
              }
              break;
            }

            case 'SwitchNextPageStatic' : {
              let uiContext: UIContext = this.getUIContext();
              let router: Router = uiContext.getRouter();
              router.pushUrl({ url: 'pages/bindSelectionMenu' })
              console.log('push URL success')
            }
          }
        })
      }

      Web({ src: 'resource://rawfile/index.html', controller: this.controller })
        .javaScriptAccess(this.javaScriptAccess)

    }
  }
}