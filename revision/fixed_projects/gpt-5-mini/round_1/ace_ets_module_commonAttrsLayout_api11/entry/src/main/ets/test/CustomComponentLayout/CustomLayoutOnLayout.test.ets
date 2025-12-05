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
import CommonFunc from '../../MainAbility/common/Common';

export default function CustomLayoutOnLayoutTest() {
  describe('CustomLayoutOnLayoutTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
     let options:router.RouterOptions = {
        url:'MainAbility/pages/CustomComponentLayout/CustomLayoutOnLayout',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get CustomLayoutOnLayout state success " + JSON.stringify(pages));
        if (!("CustomLayoutOnLayout" == pages.name)) {
          console.info("get CustomLayoutOnLayout state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push CustomLayoutOnLayout page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push CustomLayoutOnLayout page error: " + err);
      }
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("CustomLayoutOnLayoutTest after each called");
    });

    /*
     * @tc.number       SUB_ACE_BASIC_ETS_API_0001
     * @tc.name         CustomLayoutOnLayoutTest_0010
     * @tc.desic        test Rect radius
     */
    it('CustomLayoutOnLayoutTest_0010', Level.LEVEL1, async (done: Function) => {
      console.info('[CustomLayoutOnLayoutTest_0010] START');
      await CommonFunc.sleep(1000);


      let text_1 = CommonFunc.getComponentRect('CustomLayoutOnLayoutText_1');
      let text_2 = CommonFunc.getComponentRect('CustomLayoutOnLayoutText_2');
      let text_3 = CommonFunc.getComponentRect('CustomLayoutOnLayoutText_3');

      expect(Math.abs(text_2.left - text_1.left - vp2px(20)) <= 1).assertTrue();
      expect(Math.abs(text_3.left - text_2.left - vp2px(20)) <= 1).assertTrue();
      expect(Math.abs(text_2.top - text_1.top - vp2px(20)) <= 1).assertTrue();
      expect(Math.abs(text_3.top - text_2.top - vp2px(20)) <= 1).assertTrue();

      console.info('[CustomLayoutOnLayoutTest_0010] END');
      done();
    });
  })
}
