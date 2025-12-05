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

export default function SafeAreaPadding() {
  describe('SafeAreaPadding', () => {
    beforeEach(async (done: Function) => {
      console.info("SafeAreaPadding beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/BarStyle/SafeAreaPadding',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get SafeAreaPadding state pages:" + JSON.stringify(pages));
        if (!("SafeAreaPadding" == pages.name)) {
          console.info("get SafeAreaPadding pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push SafeAreaPadding page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push SafeAreaPadding page error:" + err);
      }
      console.info("SafeAreaPadding beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("SafeAreaPadding after each called");
    });


    /**
     * @tc.number    SafeAreaPadding_0100
     * @tc.name      SafeAreaPadding_0100
     * @tc.desc      Test SafeAreaPadding
     * @tc.level     Level 2
     */
    it('SafeAreaPadding_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('SafeAreaPadding_0100 START');
      await CommonFunc.sleep(1000);
      let navRect = CommonFunc.getComponentRect('BarStyle_SafeAreaPadding');
      console.info('navRect.top is' + navRect.top)
      let navContentRect = CommonFunc.getComponentRect('NavigationBarStyle_content2');
      console.info('navContentRect.top is' + navContentRect.top)
      expect(Math.abs(Math.round(navRect.top - navContentRect.top))- vp2px(112) <= 1).assertTrue();
      console.info('SafeAreaPadding_0100 END');
      done();
    });
  })
}