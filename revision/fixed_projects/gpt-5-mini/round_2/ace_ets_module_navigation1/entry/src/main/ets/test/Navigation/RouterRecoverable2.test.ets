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

export default function RouterRecoverable2() {
  describe('RouterRecoverable2', () => {
    beforeEach(async (done: Function) => {
      console.info("RouterRecoverable2 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/RouterRecoverable',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get RouterRecoverable2 state pages:" + JSON.stringify(pages));
        if (!("RouterRecoverable2" == pages.name)) {
          console.info("get RouterRecoverable2 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push RouterRecoverable2 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push RouterRecoverable2 page error:" + err);
      }
      console.info("RouterRecoverable2 beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("RouterRecoverable2 after each called");
    });


    /**
     * @tc.number    RouterRecoverable2_0100
     * @tc.name      RouterRecoverable2_0100
     * @tc.desc      Test RouterRecoverable2
     * @tc.level     Level 2
     */
    it('RouterRecoverable2_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('RouterRecoverable2_0100 START');
      await CommonFunc.sleep(200);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('routerBtnId2'));
      await btn1.click()
      console.info('[RouterRecoverable2_0100] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('RouterPage2Id');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[RouterRecoverable2_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[RouterRecoverable2_0100] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('routerPage2');

      console.info('ReplaceDestination1_0100 END');
      console.info('RouterRecoverable2_0100 END');
      done();
    });
  })
}