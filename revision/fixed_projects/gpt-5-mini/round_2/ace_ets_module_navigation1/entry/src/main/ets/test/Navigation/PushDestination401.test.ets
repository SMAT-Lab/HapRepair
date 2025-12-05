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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function PushDestination401() {
  describe('PushDestination401', () => {
    beforeEach(async (done: Function) => {
      console.info("PushDestination401 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/PushDestination401',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get PushDestination401 state pages:" + JSON.stringify(pages));
        if (!("PushDestination401" == pages.name)) {
          console.info("get PushDestination401 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push PushDestination401 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push PushDestination401 page error:" + err);
      }
      console.info("PushDestination401 beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("PushDestination401 after each called");
    });


    /**
     * @tc.number    PushDestination401_0100
     * @tc.name      PushDestination401_0100
     * @tc.desc      Test pushDestination error code 401
     * @tc.level     Level 2
     */
    it('PushDestination401_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('PushDestination401_0100 START');
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('undefinedId'));
      await btn1.click()
      console.info('[PushDestination401_0100] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('id06');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[PushDestination401_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[PushDestination401_0100] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('401');

      console.info('PushDestination401_0100 END');
      done();
    });

        /**
     * @tc.number    PushDestination401_0200
     * @tc.name      PushDestination401_0200
     * @tc.desc      Test pushDestination error code 401 with NavigationOptions
     * @tc.level     Level 2
     */
    it('PushDestination401_0200',  Level.LEVEL0, async (done: Function) => {
      console.info('PushDestination401_0200 START');
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('undefinedId'));
      await btn1.click()
      console.info('[PushDestination401_0200] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('id06');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[PushDestination401_0200] component obj is: ' + JSON.stringify(obj));
      console.info('[PushDestination401_0200] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('401');

      console.info('PushDestination401_0200 END');
      done();
    });
    
  })
}