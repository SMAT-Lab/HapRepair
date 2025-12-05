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

export default function InterceptionTest() {
  describe('InterceptionTest', () => {
    beforeEach(async (done: Function) => {
      console.info("InterceptionTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Interception/InterceptionTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get InterceptionTest state pages:" + JSON.stringify(pages));
        if (!("InterceptionTest" == pages.name)) {
          console.info("get InterceptionTest pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push InterceptionTest page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push InterceptionTest page error:" + err);
      }
      console.info("InterceptionTest beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("InterceptionTest after each called");
    });


    /**
     * @tc.number    interception_0100
     * @tc.name      interception_0100
     * @tc.desc      Test interception before willshow and implement page interception
     * @tc.level     Level 0
     */
    it('interception_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('interception_0100 START');
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('PageId'));
      await btn1.click()
      console.info('[interception_0100] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('PageTwoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[interception_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[interception_0100] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('pageOne,pageTwo');
      console.info('interception_0100 END');
      done();
    });
  })
}