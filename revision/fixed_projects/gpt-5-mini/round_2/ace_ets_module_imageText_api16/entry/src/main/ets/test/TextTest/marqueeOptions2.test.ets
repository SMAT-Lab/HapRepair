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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium";
import router from '@ohos.router';

export default function marqueeOptionsTest2() {
  describe('marqueeOptionsTest2', () => {
    beforeEach(async (done: Function) => {
      console.info("marqueeOptionsTest2 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Text/marqueeOptions2',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get marqueeOptionsTest2 state pages:" + JSON.stringify(pages));
        if (!("marqueeOptionsTest2" == pages.name)) {
          console.info("get marqueeOptionsTest2 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push marqueeOptionsTest2 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push marqueeOptionsTest2 page error:" + err);
      }
      console.info("marqueeOptionsTest2 beforeEach end");
      done();
    });
    afterEach(async () => {
    });

    /**
     * @tc.number marqueeOptionsTest2_0100
     * @tc.name testFocus
     * @tc.desc
     */
    it('marqueeOptionsTest2_0100', Level.LEVEL0, async (done: Function) => {
      console.info('marqueeOptionsTest2_0100 START');
      await CommonFunc.sleep(5000);

      let text1 = getInspectorByKey('marqueeOptionsTest2_text1');
      expect(JSON.stringify(JSON.parse(text1).$attrs.content)).assertEqual('"MarqueeState.BOUNCE"');

      let text2 = getInspectorByKey('marqueeOptionsTest2_text2');
      expect(JSON.stringify(JSON.parse(text2).$attrs.content)).assertEqual('"MarqueeState.FINISH"');
      await CommonFunc.sleep(1000);

      console.info('marqueeOptionsTest2_0100 END');
      done();
    });
  })
}