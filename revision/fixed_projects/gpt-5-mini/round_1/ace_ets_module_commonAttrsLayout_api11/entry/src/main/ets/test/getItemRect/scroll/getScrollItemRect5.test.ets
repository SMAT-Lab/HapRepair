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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import {
  UiComponent,
  UiDriver,
  Component,
  Driver,
  UiWindow,
  ON,
  BY,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';
import CommonFunc from '../../../MainAbility/common/Common';
import { MessageManager, Callback } from '../../../MainAbility/common/MessageManager';
import ohosrouter from '@ohos.router';
import { BusinessError } from '@ohos.base';



export default function getScrollItemRect5() {
  describe('getScrollItemRect5', () => {
    beforeEach(async (done: Function) => {
      console.info("getScrollItemRect5 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/getItemRect/scroll/getScrollItemRect5',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get getScrollItemRect5 state pages: " + JSON.stringify(pages));
        if (!("getScrollItemRect5" == pages.name)) {
          console.info("get getScrollItemRect5 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push getScrollItemRect5 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getScrollItemRect5 page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getScrollItemRect5_0100
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 0,set padding of scroll to 20
    * @tc.level : Level 2
    */
    it('getScrollItemRect5_0100', Level.LEVEL0, async (done: Function) => {

      console.info('[getScrollItemRect5_0100] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'padding1', value: 20
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getScrollItemRect5_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getScrollItemRect5_text1');
      let text2 = CommonFunc.getComponentRect('getScrollItemRect5_text2');
      let text3 = CommonFunc.getComponentRect('getScrollItemRect5_text3');
      let text4 = CommonFunc.getComponentRect('getScrollItemRect5_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getScrollItemRect5_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getScrollItemRect5_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getScrollItemRect5_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getScrollItemRect5_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(20));
      expect(Math.round(y)).assertEqual(Math.round(20));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(300));
      expect(Math.round(height)).assertEqual(Math.round(600));

      done();
    });


    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getScrollItemRect5_0200
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 0,set margin of column to 20
    * @tc.level : Level 2
    */
    it('getScrollItemRect5_0200', Level.LEVEL0, async (done: Function) => {

      console.info('[getScrollItemRect5_0200] START');

      await CommonFunc.sleep(1000);
      try {
        ohosrouter.back({
          url: 'MainAbility/pages/index/index'
        });
        await CommonFunc.sleep(500);
        ohosrouter.clear();
        await CommonFunc.sleep(500);
        let pages = ohosrouter.getState();
        console.info("getScrollItemRect5_0200 " + JSON.stringify(pages));
        ohosrouter.pushUrl({
          url: 'MainAbility/pages/getItemRect/scroll/getScrollItemRect5',
          params: {
            data1: 'message',
            data2: {
              data3: [123, 456, 789]
            }
          }
        }).then(() => {
          // success
        })
          .catch((err: BusinessError) => {
            console.error(`pushUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        await CommonFunc.sleep(2000);
      } catch (err) {
        console.error("push getScrollItemRect5 page error: " + err);
      }

      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      globalThis.value.message.notify({
        name: 'margin1', value: 20
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getScrollItemRect5_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getScrollItemRect5_text1');
      let text2 = CommonFunc.getComponentRect('getScrollItemRect5_text2');
      let text3 = CommonFunc.getComponentRect('getScrollItemRect5_text3');
      let text4 = CommonFunc.getComponentRect('getScrollItemRect5_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getScrollItemRect5_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getScrollItemRect5_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getScrollItemRect5_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getScrollItemRect5_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(20));
      expect(Math.round(y)).assertEqual(Math.round(20));

      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(300));
      expect(Math.round(height)).assertEqual(Math.round(600));

      done();
    });


  })
}
