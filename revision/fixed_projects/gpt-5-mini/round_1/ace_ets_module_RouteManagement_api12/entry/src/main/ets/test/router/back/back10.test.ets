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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import CommonFunc from '../../../MainAbility/utils/Common';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function back10() {
  describe('back10', () => {
    beforeEach(async (done: Function) => {
      console.info("back10 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/back/back10',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get back10 state success " + JSON.stringify(pages));
        if (!("back10" == pages.name)) {
          console.info("get back10 state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push back10 page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push back10 page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("back10 after each called");
    });

    /**
     * @tc.number    back10_0100
     * @tc.name      back10_0100
     * @tc.desc      Test back
     * @tc.level     Level 2
     */
    it('back10_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[back10_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.id('back10_btn'));
      await btn1.click();
      await CommonFunc.sleep(2000);
      console.info('[back10_0100] btn1');

      let btn2 = await driver.findComponent(ON.id('back010_btn'));
      await btn2.click();
      await CommonFunc.sleep(2000);
      console.info('[back10_0100] btn2');

      let text1 = getInspectorByKey('back10_text');
      let Obj1: Record<string, ESObject> = JSON.parse(text1);
      console.info("[back10_0100] component obj is: " + JSON.stringify(Obj1));
      expect(Obj1.$attrs.content).assertEqual('back10');

      console.info('[back10_0100] END');
      done();
    });

  })
}
