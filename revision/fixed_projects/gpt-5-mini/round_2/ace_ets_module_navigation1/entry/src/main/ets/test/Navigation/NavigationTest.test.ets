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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function NavigationTest() {
  describe('NavigationTest',  () =>{
    beforeEach(async  (done:Function) =>{
      console.info("NavigationTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/NavigationTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationTest state success " + JSON.stringify(pages));
        if (pages.name !== 'NavigationTest') { 
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationTest page error: " + err);
      }
      done()
    });

    afterEach(async  () =>{
      await CommonFunc.sleep(1000);
      console.info("NavigationTest after each called");
    });

    /*
     * @tc.number      NavigationMode_001
     * @tc.name        Navigation basic navigation function test
     * @tc.desc        Test Navigation push and pop functions between pageOne001 and pageTwo002
     * @tc.level       Level 2
     */
    it('NavigationMode_001',  Level.LEVEL2, async  (done:Function) =>{
      console.info('[NavigationMode_001] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let pushPageOne001Btn = await driver.findComponent(ON.id('pageTwo002'));
      await pushPageOne001Btn.click();
      await CommonFunc.sleep(2000);
      console.info("[NavigationMode_001] Jumped to pageTwo002");

      let pageTwo002ToPageOne001Btn = await driver.findComponent(ON.id('pageOne01'));
      await pageTwo002ToPageOne001Btn.click();
      await CommonFunc.sleep(2000);
      console.info("[NavigationMode_001] Jumped back to pageOne001 from pageTwo002");

      let pushPageTwo002Btn = await driver.findComponent(ON.id('pageTwo02'));
      await pushPageTwo002Btn.click();
      await CommonFunc.sleep(2000);
      console.info("[NavigationMode_001] Jumped to pageTwo002");

      let strJson = getInspectorByKey('indexLogId');
      let obj: ESObject = JSON.parse(strJson);
      console.info('[NavigationMode_001] obj.$attrs.content is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('didShow:from.mode 0to.mode 0');

      console.info('[NavigationMode_001] END');
      done();
    });
  })
}