/**
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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import CommonFunc from '../../MainAbility/utils/Common';
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

export default function router_getStateByIndex4() {
  describe('router_getStateByIndex4', () => {
    beforeEach(async (done: Function) => {
      console.info("router_getStateByIndex4 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/router_getStateByIndex4',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get router_getStateByIndex4 state pages:" + JSON.stringify(pages));
        if (!("router_getStateByIndex4" == pages.name)) {
          console.info("get router_getStateByIndex4 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push router_getStateByIndex4 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push router_getStateByIndex4 page error:" + err);
      }
      console.info("router_getStateByIndex4 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("router_getStateByIndex4 after each called");
    });

    /**
     * @tc.number SUB_ACE_UI_ABILITY_ROUTER_0040
     * @tc.name testGetStateByIndex
     * @tc.desc Retrieve the status information of the corresponding page through index values.
     * @tc.level Level 3
     * @tc.type
     * @tc.size
     */
    it('router_getStateByIndex4_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[router_getStateByIndex4_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.id('router_getStateByIndex4_btn1'));
      await btn1.click();
      await CommonFunc.sleep(2000);
      console.info('[router_getStateByIndex4_0100] btn1');

      let text1 = getInspectorByKey('router_getStateByIndex4_text1');
      expect(JSON.parse(text1).$attrs.content).assertEqual('undefined');
      console.info('[router_getStateByIndex4_0100] text1');

      let btn2 = await driver.findComponent(ON.id('router_getStateByIndex4_btn2'));
      await btn2.click();
      await CommonFunc.sleep(2000);
      console.info('[router_getStateByIndex4_0100] btn2');

      let text2 = getInspectorByKey('router_getStateByIndex4_text2');
      expect(JSON.parse(text2).$attrs.content).assertEqual('undefined');
      console.info('[router_getStateByIndex4_0100] text2');

      let btn3 = await driver.findComponent(ON.id('router_getStateByIndex4_btn3'));
      await btn3.click();
      await CommonFunc.sleep(2000);
      console.info('[router_getStateByIndex4_0100] btn3');

      let text3 = getInspectorByKey('router_getStateByIndex4_text3');
      expect(JSON.parse(text3).$attrs.content).assertEqual('undefined');
      console.info('[router_getStateByIndex4_0100] text3');

      let btn4 = await driver.findComponent(ON.id('router_getStateByIndex4_btn4'));
      await btn4.click();
      await CommonFunc.sleep(2000);
      console.info('[router_getStateByIndex4_0100] btn4');

      let text4 = getInspectorByKey('router_getStateByIndex4_text4');
      expect(JSON.parse(text4).$attrs.content).assertEqual('undefined');
      console.info('[router_getStateByIndex4_0100] text4');

      console.info('[router_getStateByIndex4_0100] END');
      done();
    });

  })
}
