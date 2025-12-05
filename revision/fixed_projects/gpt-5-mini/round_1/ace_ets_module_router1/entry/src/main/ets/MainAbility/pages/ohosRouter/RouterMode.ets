/**
 * Copyright (c) 2025 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { BusinessError } from '@kit.BasicServicesKit';
import router from '@ohos.router'

class innerParams {
  data3: number[];

  constructor(tuple: number[]) {
    this.data3 = tuple;
  }
}

class routerParams {
  data1: string;
  data2: innerParams;

  constructor(str: string, tuple: number[]) {
    this.data1 = str;
    this.data2 = new innerParams(tuple);
  }
}

@Entry
@Component
struct RouterMode {
  build() {
    Flex({ justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
      Column() {
        Button('routerPushUrl401')
          .id('401Btn1')
          .height('10%')
          .onClick(()=>{
            try {
              router.pushUrl({
                url: undefined,
                params: new routerParams('message' ,[123,456,789]),
                recoverable: true,
              }, router.RouterMode.Single)
                .catch((err: ESObject) => {
                  if (err && err?.code) {
                    console.error(`pushUrl failed, code is ${(err as BusinessError)?.code}, message is ${(err as BusinessError)?.message}`);
                  }
                })
            } catch (err) {
              if (err && err?.code) {
                console.error(`pushUrl1 failed, code is ${err?.code}, message is ${err?.message}`)
                AppStorage.setOrCreate('401Code', err.code);
              }
            }
          })

        Button('routerPushUrl100002')
          .id('100002Btn1')
          .height('10%')
          .onClick(() => {
          router.pushUrl({
            url: 'pages/routerPage2',
            params: new routerParams('message', [123, 456, 789]),
            recoverable: false
          }, router.RouterMode.Standard)
            .then(() => {
              console.error(`pushUrl finish`);
            })
            .catch((err: ESObject) => {
              if (err && err?.code) {
                console.error(`pushUrl failed, code is ${(err as BusinessError)?.code}, message is ${(err as BusinessError)?.message}`);
                AppStorage.setOrCreate('100002Code', (err as BusinessError)?.code);
                } else {
                  console.error(`pushUrl1 failed, err is ${err}`)
              }
            })
        })

        Button('routerPushUrl100003')
          .id('100003Btn1')
          .height('10%')
          .onClick(() => {
          for(let i =0;i<32;i++){
            router.pushUrl({
              url: 'MainAbility/pages/ohosRouter/router',
              params: new routerParams('message', [123, 456, 789]),
              recoverable: true
            }, router.RouterMode.Standard)
              .then(() => {
                console.error(`pushUrl finish`);
              })
              .catch((err: ESObject) => {
                if (err && err?.code) {
                  console.error(`pushUrl failed, code is ${(err as BusinessError)?.code}, message is ${(err as BusinessError)?.message}`);
                  AppStorage.setOrCreate('100003Code', (err as BusinessError)?.code);
                } else {
                  console.error(`pushUrl1 failed, err is ${err}`)
                }
              })
          }
        })
          .id('100003Btn2')
          .height('10%')
          .onClick(()=>{
            for(let i =0;i<32;i++){
              router.pushUrl({
                url: 'MainAbility/pages/ohosRouter/router',
                params: new routerParams('message', [123, 456, 789])
              }, router.RouterMode.Standard, (err) => {
                if (err && err?.code) {
                  console.error(`pushUrl failed, code is ${err?.code}, message is ${err?.message}`);
                  AppStorage.setOrCreate('100003Code', err?.code);
                  return;
                } else {
                  console.error(`pushUrl1 failed, err is ${err}`)
                }
              })
            }
          })
      }
    }
  }
}
