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

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import { Driver, ON} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common'

export default function preventDefault() {
  describe('preventDefault', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Hyperlink/preventDefault',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get preventDefault state success " + JSON.stringify(pages));
        if (!("preventDefault" == pages.name)) {
          console.info("get preventDefault state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push preventDefault page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push preventDefault page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("preventDefault after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         preventDefault0001
     * @tc.desic        Intercepting Hyperlinks
     */
    it('preventDefault0001', Level.LEVEL0, async () => {
        let driver = Driver.create();
        console.info('preventDefault preventDefault0001 START');
        await CommonFunc.sleep(1000);
        let hyperlink1 = await driver.findComponent(ON.id('hyperlink1'));
        await CommonFunc.sleep(500);
        await hyperlink1.click()
        await CommonFunc.sleep(500);

        let hyperlink3 = await driver.findComponent(ON.id('hyperlink3'));
        await CommonFunc.sleep(500);
        await hyperlink3.click()
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('禁掉onclick'));
        console.info("[preventDefault0001] end");
    });
  })
}
