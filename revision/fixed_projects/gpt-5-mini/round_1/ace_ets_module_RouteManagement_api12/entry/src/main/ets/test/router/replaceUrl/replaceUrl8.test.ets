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


export default function replaceUrl8() {
  describe('replaceUrl8', () => {
    beforeEach(async (done: Function) => {
      console.info("replaceUrl8 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/replaceUrl/replaceUrl8',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get replaceUrl8 state success " + JSON.stringify(pages));
        if (!("replaceUrl8" == pages.name)) {
          console.info("get replaceUrl8 state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push replaceUrl8 page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push replaceUrl8 page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("replaceUrl8 after each called");
    });

    /**
     * @tc.number    replaceUrl8_0100
     * @tc.name      replaceUrl8_0100
     * @tc.desc      Test replaceUrl
     * @tc.level     Level 2
     */
    it('replaceUrl8_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[replaceUrl8_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn = await driver.findComponent(ON.id('replaceUrl8_btn'));
      await btn.click()
      await CommonFunc.sleep(2000);
      console.info('[replaceUrl8_0100] btn click');

      let btn2 = await driver.findComponent(ON.id('replaceUrl8_get'));
      await btn2.click()
      await CommonFunc.sleep(1000);
      console.info('[replaceUrl8_0100] btn2 click');

      let strJson = getInspectorByKey('replaceUrl8_text2');
      await CommonFunc.sleep(1000);
      console.info("[replaceUrl8_0100] component obj is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('undefined;replaceUrl8_aboutToAppear;replaceUrl8_aboutToAppear');

      console.info('[replaceUrl8_0100] END');
      done();
    });
  })
}

