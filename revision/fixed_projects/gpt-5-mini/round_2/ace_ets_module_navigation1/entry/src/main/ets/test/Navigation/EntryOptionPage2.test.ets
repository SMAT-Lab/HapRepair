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

export default function EntryOptionPage2() {
  describe('EntryOptionPage2', () => {
    beforeEach(async (done: Function) => {
      console.info("EntryOptionPage2 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/EntryOptionPage2',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get EntryOptionPage2 state pages:" + JSON.stringify(pages));
        if (!("EntryOptionPage2" == pages.name)) {
          console.info("get EntryOptionPage2 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push EntryOptionPage2 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push EntryOptionPage2 page error:" + err);
      }
      console.info("EntryOptionPage2 beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("EntryOptionPage2 after each called");
    });


    /**
     * @tc.number    EntryOptionPage2_0100
     * @tc.name      EntryOptionPage2_0100
     * @tc.desc      Test EntryOptionPage2 useSharedStorage is true
     * @tc.level     Level 0
     */
    it('EntryOptionPage2_0100', Level.LEVEL0, async (done: Function) => {
      console.info('EntryOptionPage2_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(2000);

      let strJson: string = getInspectorByKey('text2Id');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[MultiNavigation_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[MultiNavigation_0100] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('333');

      console.info('EntryOptionPage2_0100 END');
      done();
    });

  })
}