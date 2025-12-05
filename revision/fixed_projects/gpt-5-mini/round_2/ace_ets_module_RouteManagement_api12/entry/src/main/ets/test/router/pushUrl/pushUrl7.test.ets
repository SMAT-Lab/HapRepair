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

import CommonFunc from '../../../MainAbility/utils/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function pushUrl7() {
  describe('pushUrl7', () => {
    beforeEach(async (done: Function) => {
      console.info("pushUrl7 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/pushUrl/pushUrl7',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get pushUrl7 state pages:" + JSON.stringify(pages));
        if (!("List_SpaceChange" == pages.name)) {
          console.info("get pushUrl7 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push pushUrl7 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push pushUrl7 page error:" + err);
      }
      console.info("pushUrl7 beforeEach end");
      done();
    });
    afterEach(async () => {
    });

    /**
     * @tc.number pushUrl7_0100
     * @tc.name testPushUrl
     * @tc.desc
     */
    it('pushUrl7_0100', Level.LEVEL2, async (done: Function) => {
      console.info('pushUrl7_0100 START');
      await CommonFunc.sleep(1000);

      sendEventByKey('pushUrl7_btn1',10,"")
      await CommonFunc.sleep(1000);
      sendEventByKey('pushUrl7_btn1',10,"")
      await CommonFunc.sleep(1000);

      let text1 = getInspectorByKey('pushUrl7_text2');
      expect(String(JSON.parse(text1).$attrs.content)).assertEqual('100003');
      await CommonFunc.sleep(1000);

      console.info('pushUrl7_0100 END');
      done();
    });

  })
}