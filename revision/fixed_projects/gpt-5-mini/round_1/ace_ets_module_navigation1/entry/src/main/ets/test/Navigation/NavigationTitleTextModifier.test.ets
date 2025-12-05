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

export default function NavigationTitleTextModifier() {
  describe('NavigationTitleTextModifier', () => {
    beforeEach(async (done: Function) => {
      console.info("NavigationTitleTextModifier beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/NavigationTitleTextModifier',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationTitleTextModifier state pages:" + JSON.stringify(pages));
        if (!("NavigationTitleTextModifier" == pages.name)) {
          console.info("get NavigationTitleTextModifier pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationTitleTextModifier page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationTitleTextModifier page error:" + err);
      }
      console.info("NavigationTitleTextModifier beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavigationTitleTextModifier after each called");
    });


    /**
     * @tc.number    NavigationTitleTextModifier_0100
     * @tc.name      NavigationTitleTextModifier_0100
     * @tc.desc      Test NavigationTitleTextModifier
     * @tc.level     Level 2
     */
    it('NavigationTitleTextModifier_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('NavigationTitleTextModifier_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.id('applyModifier_toggleId'));
      await btn1.click()
      console.info('[NavigationTitleTextModifier_0100] btn1 click');
      await CommonFunc.sleep(1000);

      let obj1: ESObject = JSON.parse(JSON.stringify(getInspectorTree()));
      await CommonFunc.sleep(1000);

      expect(obj1.$children[0].$children[0].$children[0].$children[0].$attrs.fontColor).assertEqual("#FFFFC000");
      console.info("[NavigationTitleTextModifier_0100] main title equal: " + '#FFFFC000');
      expect(obj1.$children[0].$children[0].$children[0].$children[1].$attrs.fontColor).assertEqual("#FFFFCEEE");
      console.info("[NavigationTitleTextModifier_0100] sub title equal: " + '#FFFFCEEE');

      console.info('NavigationTitleTextModifier_0100 END');
      done();
    });


  })
}