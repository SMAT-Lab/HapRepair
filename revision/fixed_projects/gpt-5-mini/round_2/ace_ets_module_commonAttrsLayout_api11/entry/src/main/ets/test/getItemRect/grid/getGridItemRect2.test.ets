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


export default function getGridItemRect2() {
  describe('getGridItemRect2', () => {
    beforeEach(async (done: Function) => {
      console.info("getGridItemRect2 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/getItemRect/grid/getGridItemRect2',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get getGridItemRect2 state pages: " + JSON.stringify(pages));
        if (!("getGridItemRect2" == pages.name)) {
          console.info("get getGridItemRect2 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push getGridItemRect2 page result: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getGridItemRect2 page error: " + err);
        expect().assertFail();
      }
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back({
        url: 'MainAbility/pages/index/index'
      })
      console.info("BadgeFocus after each called");
    });
    /* *
    * @tc.number: SUB_ACE_TS_COMPONENT_getGridItemRect2_0100
    * @tc.name  : onGetItemRect
    * @tc.desc  : set index to 3
    * @tc.level : Level 2
    */
    it('getGridItemRect2_0100', Level.LEVEL0, async (done: Function) => {

      console.info('[getGridItemRect2_0100] START');
      globalThis.value.message.notify({
        name: 'index', value: 0
      });
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      console.log('button2 click')
      let btn2 = await driver.findComponent(ON.id('getGridItemRect2_button2'));
      await btn2.click();
      await CommonFunc.sleep(2000);

      console.log('button click')
      let btn = await driver.findComponent(ON.id('getGridItemRect2_button'));
      await btn.click();
      await CommonFunc.sleep(2000);

      console.log('get text value')
      let text1 = CommonFunc.getComponentRect('getGridItemRect2_text1');
      let text2 = CommonFunc.getComponentRect('getGridItemRect2_text2');
      let text3 = CommonFunc.getComponentRect('getGridItemRect2_text3');
      let text4 = CommonFunc.getComponentRect('getGridItemRect2_text4');

      console.log('get text message')
      let strJson1 = getInspectorByKey('getGridItemRect2_text1')
      let x = Number(JSON.parse(strJson1).$attrs.content)
      console.log('x' + x)

      let strJson2 = getInspectorByKey('getGridItemRect2_text2')
      let y = Number(JSON.parse(strJson2).$attrs.content)
      console.log('y' + y)

      let strJson3 = getInspectorByKey('getGridItemRect2_text3')
      let width = Number(JSON.parse(strJson3).$attrs.content)
      console.log('width' + width)

      let strJson4 = getInspectorByKey('getGridItemRect2_text4')
      let height = Number(JSON.parse(strJson4).$attrs.content)
      console.log('height' + height)

      console.log('assert position')
      expect(Math.round(x)).assertEqual(Math.round(0));
      expect(Math.round(y)).assertEqual(Math.round(0));
      console.log('assert size')
      expect(Math.round(width)).assertEqual(Math.round(0));
      expect(Math.round(height)).assertEqual(Math.round(0));

      done();
    });

  })
}
