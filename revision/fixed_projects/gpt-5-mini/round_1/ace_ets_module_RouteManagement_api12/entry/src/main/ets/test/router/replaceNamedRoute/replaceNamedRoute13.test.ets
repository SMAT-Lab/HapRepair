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

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import CommonFunc from '../../../MainAbility/utils/Common';
import { Driver, ON } from '@ohos.UiTest';


export default function replaceNamedRoute13() {
  describe('replaceNamedRoute13', () => {
    beforeEach(async (done: Function) => {
      console.info("replaceNamedRoute13 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/replaceNamedRoute/replaceNamedRoute13',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get replaceNamedRoute13 state success " + JSON.stringify(pages));
        if (!("replaceNamedRoute13" == pages.name)) {
          console.info("get replaceNamedRoute13 state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push replaceNamedRoute13 page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push replaceNamedRoute13 page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("replaceNamedRoute13 after each called");
    });

    /**
     * @tc.number    replaceNamedRoute13_0100
     * @tc.name      replaceNamedRoute13_0100
     * @tc.desc      Test replaceNamedRoute
     * @tc.level     Level 2
     */
    it('replaceNamedRoute13_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[replaceNamedRoute13_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn = await driver.findComponent(ON.id('replaceNamedRoute13_btn'));
      await btn.click()
      await CommonFunc.sleep(2000);
      console.info('[replaceNamedRoute13_0100] btn click');

      let btn2 = await driver.findComponent(ON.id('nameDetail13_btn'));
      await btn2.click()
      await CommonFunc.sleep(2000);
      console.info('[replaceNamedRoute13_0100] btn2 click');

      let strJson = getInspectorByKey('nameDetail13_text');
      await CommonFunc.sleep(1000);
      console.info("[replaceNamedRoute13_0100] component obj is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('nameDetail13 index2');

      console.info('[replaceNamedRoute13_0100] END');
      done();
    });
  })
}

