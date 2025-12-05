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

export default function pushNamedRoute3() {
  describe('pushNamedRoute3', () => {
    beforeEach(async (done: Function) => {
      console.info("pushNamedRoute3 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/pushNamedRoute/pushNamedRoute3',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get pushNamedRoute3 state pages:" + JSON.stringify(pages));
        if (!("List_SpaceChange" == pages.name)) {
          console.info("get pushNamedRoute3 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push pushNamedRoute3 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push pushNamedRoute3 page error:" + err);
      }
      console.info("pushNamedRoute3 beforeEach end");
      done();
    });
    afterEach(async () => {
    });

    /**
     * @tc.number pushNamedRoute3_0100
     * @tc.name testButton
     * @tc.desc
     */
    it('pushNamedRoute3_0100', Level.LEVEL2, async (done: Function) => {
      console.info('pushNamedRoute3_0100 START');
      await CommonFunc.sleep(1000);

      sendEventByKey('pushNamedRoute3_btn1',10,"")
      await CommonFunc.sleep(5000);
      sendEventByKey('pushNamedRoute3_btn1',10,"")
      await CommonFunc.sleep(2000);
      let text1 = getInspectorByKey('pushNamedRoute3_text2');
      expect(String(JSON.parse(text1).$attrs.content)).assertEqual('100003');
      await CommonFunc.sleep(1000);

      console.info('pushNamedRoute3_0100 END');
      done();
    });

  })
}