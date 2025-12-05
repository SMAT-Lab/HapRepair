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
import {
  UiComponent,
  Driver,
  Component,
  UiWindow,
  ON,
  BY,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/utils/Common';
import { MessageManager, Callback } from '../../MainAbility/utils/MessageManager';


export default function pushPageInit() {
  describe('pushPageInit', () => {
    beforeEach(async (done: Function) => {
      console.info("pushPageInit beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/supplementRouter/pushPageInit',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get pushPageInit state pages: " + JSON.stringify(pages));
        if (!("pushPageInit" == pages.name)) {
          console.info("get pushPageInit state pages.name: " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push pushPageInit success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push pushPageInit page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /**
     * @tc.number SUB_ACE_BASICABILITY_ROUTERPAGE_002
     * @tc.name pushPageInit_0100
     * @tc.desc set api
     * @tc.level: Level 2
     * @tc.type :
     * @tc.size :
     */
    it('pushPageInit_0100', Level.LEVEL2, async (done: Function) => {
      console.info('pushPageInit_0100 START');
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      let button1 = await driver.findComponent(ON.id('pushPageInit'));
      await button1.click();
      await CommonFunc.sleep(1000);

      let strJson1 = getInspectorByKey('pushPageA_amount');
      let strJson2 = getInspectorByKey('pushPageA_page');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('3');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('3');

      let button2 = await driver.findComponent(ON.id('pushPageA'));
      await button2.click();
      await CommonFunc.sleep(1000);

      let strJson3 = getInspectorByKey('pushPageB_amount');
      let strJson4 = getInspectorByKey('pushPageB_page');
      expect(JSON.parse(strJson3).$attrs.content).assertEqual('4');
      expect(JSON.parse(strJson4).$attrs.content).assertEqual('4');
      let button3 = await driver.findComponent(ON.id('pushPageB'));
      await button3.click();
      await CommonFunc.sleep(1000);

      let strJson5 = getInspectorByKey('pushPageC_amount');
      let strJson6 = getInspectorByKey('pushPageC_page');
      expect(JSON.parse(strJson5).$attrs.content).assertEqual('5');
      expect(JSON.parse(strJson6).$attrs.content).assertEqual('5');
      let button4 = await driver.findComponent(ON.id('pushPageC'));
      await button4.click();
      await CommonFunc.sleep(1000);

      let strJson7 = getInspectorByKey('pushPageA_amount');
      let strJson8 = getInspectorByKey('pushPageA_page');
      expect(JSON.parse(strJson7).$attrs.content).assertEqual('3');
      expect(JSON.parse(strJson8).$attrs.content).assertEqual('3');
      await CommonFunc.sleep(1000);
      console.info('[pushPageInit_0100] END');
      done();
    });
  })
}

