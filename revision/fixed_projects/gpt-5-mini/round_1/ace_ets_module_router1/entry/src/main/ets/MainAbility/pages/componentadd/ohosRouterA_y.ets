/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
import router from '@ohos.router';

@Entry
@Component
struct OhosRouterA {
  build() {
    Flex({ direction: FlexDirection.Column, alignItems: ItemAlign.Center, justifyContent: FlexAlign.Center }) {
      Text('A Page')
        .fontSize(50)
        .fontWeight(FontWeight.Bold)
      Button('Push')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          router.push({
            url: 'pages/ohosRouterB',
            params: {
              data: 'B Page'
            }
          });
        });
      Button('back')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          router.push({
            url: 'pages/ohosRouterB',
            params: {
              data: 'B Page'
            }
          });
        });
      Button('pushUrl')
        .key('pushButtonA1')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          try {
            router.pushUrl({
              url: 'pages/routerpage2',
              params: {
                data1: 'message',
                data2: {
                  data3: [123, 456, 789]
                },
              },
            })
              .then(() => {
                // success
              })
              .catch(err => {
                console.error(`pushUrl failed, code is ${err.code}, message is ${(err as Error)?.message}`);
              })
          } catch (error) {
            console.error(`pushUrl args error code is ${(error as BusinessError)?.code}, message is ${(error as Error)?.message}`);
          }
          ;
        });
      Button('pushUrl')
        .key('pushButtonA1')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          try {
            router.pushUrl({
              url: 'pages/routerpage2',
              params: {
                data1: 'message',
                data2: {
                  data3: [123, 456, 789]
                },
              },
            }, router.RouterMode.Standard)
              .then(() => {
                // success
              })
              .catch(err => {
                console.error(`pushUrl failed, code is ${err.code}, message is ${(err as Error)?.message}`);
              })
          } catch (error) {
            console.error(`pushUrl args error code is ${(error as BusinessError)?.code}, message is ${(error as Error)?.message}`);
          }
          ;
        });

      Button('replaceUr2')
        .key('pushButtonA2')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          try {
            router.replaceUrl({
              url: 'pages/detail',
              params: {
                data1: 'message',
              },
            }, router.RouterMode.Standard)
              .then(() => {
                // success
              })
              .catch(err => {
                console.error(`replaceUrl failed, code is ${err.code}, message is ${(err as Error)?.message}`);
              })
          } catch (error) {
            console.error(`replaceUrl args error code is ${(error as BusinessError)?.code}, message is ${(error as Error)?.message}`);
          }
          ;
        });
      Button('replaceUrl')
        .key('pushButtonA1')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          try {
            router.replaceUrl({
              url: 'pages/detail',
              params: {
                data1: 'message',
              },
            })
              .then(() => {
                // success
              })
              .catch(err => {
                console.info(`replaceUrl failed, code is ${err.code}, message is ${(err as Error)?.message}`);
              })
          } catch (error) {
            console.info(`replaceUrl args error code is ${(error as BusinessError)?.code}, message is ${(error as Error)?.message}`);
          }
          ;
        });

      Button('clear')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          router.push({
            uri: 'pages/ohosRouterB',
            params: {
              data: 'B Page'
            }
          });
        });
      Button('getLength')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          router.push({
            url: 'pages/ohosRouterB',
            params: {
              data: 'B Page'
            }
          });
        });
      Button('enableAlertBeforeBackPage')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          router.push({
            url: 'pages/ohosRouterB',
            params: {
              data: 'B Page'
            }
          });
        });
      Button('enableBackPageAlert')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          try {
            router.enableBackPageAlert({
              message: 'Message Info'
            });
          } catch (error) {
            console.error(`enableBackPageAlert failed, code is ${(error as BusinessError)?.code}, message is ${(error as Error)?.message}`);
          }
        });

      Button('disableAlertBeforeBackPage')
        .key('pushButtonA')
        .backgroundColor('#FF0B8FE2')
        .fontSize(20)
        .margin({ top: 20 })
        .onClick(() => {
          router.push({
            url: 'pages/ohosRouterB',
            params: {
              data: 'B Page'
            }
          });
        });
    }.width('100%').height('100%')
  }
}