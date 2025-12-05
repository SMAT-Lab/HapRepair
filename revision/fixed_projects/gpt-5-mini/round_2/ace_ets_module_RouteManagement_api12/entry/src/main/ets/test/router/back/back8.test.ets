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

export default function back8() {
  describe('back8', () => {
    beforeEach(async (done: Function) => {
      console.info("back8 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/back/back8',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get back8 state success " + JSON.stringify(pages));
        if (!("back8" == pages.name)) {
          console.info("get back8 state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push back8 page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push back8 page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("back8 after each called");
    });

    /**
     * @tc.number    back8_0100
     * @tc.name      back8_0100
     * @tc.desc      Test back
     * @tc.level     Level 2
     */
    it('back8_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[back8_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.id('back8_btn'));
      await btn1.click();
      await CommonFunc.sleep(2000);
      console.info('[back8_0100] btn1');

      let btn2 = await driver.findComponent(ON.id('back08_btn'));
      await btn2.click();
      await CommonFunc.sleep(2000);
      console.info('[back8_0100] btn2');

      await driver.pressBack()
      await CommonFunc.sleep(2000);
      console.info('[back8_0100] pressBack');

      let btn3 = await driver.findComponent(ON.text('确定'));
      await btn3.click();
      await CommonFunc.sleep(2000);
      console.info('[back8_0100] btn3');

      let text1 = getInspectorByKey('back8_text');
      let Obj1: Record<string, ESObject> = JSON.parse(text1);
      console.info("[back8_0100] component obj is: " + JSON.stringify(Obj1));
      expect(Obj1.$attrs.content).assertEqual('back8');

      console.info('[back8_0100] END');
      done();
    });

  })
}
