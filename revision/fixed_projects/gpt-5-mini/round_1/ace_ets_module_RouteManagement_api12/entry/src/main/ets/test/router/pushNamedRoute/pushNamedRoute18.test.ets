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


export default function pushNamedRoute18() {
  describe('pushNamedRoute18', () => {
    beforeEach(async (done: Function) => {
      console.info("pushNamedRoute18 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/pushNamedRoute/pushNamedRoute18',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get pushNamedRoute18 state pages:" + JSON.stringify(pages));
        if (!("pushNamedRoute18" == pages.name)) {
          console.info("get pushNamedRoute18 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push pushNamedRoute18 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push pushNamedRoute18 page error:" + err);
      }
      console.info("pushNamedRoute18 beforeEach end");
      done();
    });
    afterEach(async () => {
    });

    /**
     * @tc.number pushNamedRoute18_0100
     * @tc.name testRouter
     * @tc.desc
     */
    it('pushNamedRoute18_0100', Level.LEVEL2, async (done: Function) => {
      console.info('pushNamedRoute18_0100 START');
      await CommonFunc.sleep(1000);

      sendEventByKey('pushNamedRoute18_btn1',10,"")
      await CommonFunc.sleep(2000);

      let text1 = getInspectorByKey('pageOne_text1');
      expect(String(JSON.parse(text1).$attrs.content)).assertEqual('pageOne');
      await CommonFunc.sleep(1000);

      console.info('pushNamedRoute18_0100 END');
      done();
    });

  })
}