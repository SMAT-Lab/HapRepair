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
struct RouterCall {
  build() {
    Flex({ justifyContent: FlexAlign.Center, alignItems: ItemAlign.Center }) {
      Column() {
        Button('routerCallbackPush401')
          .height('10%')
          .id('401Btn2')
          .onClick(()=>{
            try {
              router.pushUrl({
                url: undefined,
                params: new routerParams('message' ,[123,456,789]),
                recoverable: true,
              }, (err) => {
                if (err) {
                  console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
                  return;
                }
              })
            } catch (err) {
              console.error(`pushUrl1 failed, code is ${err.code}, message is ${err.message}`)
              AppStorage.setOrCreate('401Code', err.code);
            }
          })

        Button('routerCallbackPush100002')
          .id('100002Btn2')
          .height('10%')
          .onClick(() => {
            router.pushUrl({
              url: 'pages/routerPage2',
              params: new routerParams('message', [123, 456, 789]),
              recoverable: false
            }, (err) => {
              if (err) {
                console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
                AppStorage.setOrCreate('100002Code', err.code);
                return;
              }
            })
          })

        Button('routerCallbackPush100003')
          .id('100003Btn2')
          .height('10%')
          .onClick(()=>{
            for(let i =0;i<32;i++){
              router.pushUrl({
                url: 'MainAbility/pages/ohosRouter/router',
                params: new routerParams('message', [123, 456, 789])
              }, (err) => {
                if (err) {
                  console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
                  AppStorage.setOrCreate('100003Code', err.code);
                  return;
                }})
            }
          })
      }
    }
  }
}
