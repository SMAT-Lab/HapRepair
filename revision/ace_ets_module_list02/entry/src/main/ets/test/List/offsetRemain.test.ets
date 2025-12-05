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
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function offsetRemain() {
  describe('offsetRemain', () => {
    beforeEach(async (done: Function) => {
      console.info("offsetRemain beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/offsetRemain",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get offsetRemain state pages: " + JSON.stringify(pages));
        if (!("offsetRemain" == pages.name)) {
          console.info("get offsetRemain state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push offsetRemain success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push offsetRemain page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /**
     * @tc.number    offsetRemain_0100
     * @tc.name      offsetRemain_0100
     * @tc.desc      frameNode: offsetRemainToWindow
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('offsetRemain_0100', Level.LEVEL2, async (done: Function) => {
      console.info('offsetRemain_0100 START');
      let driver = Driver.create();
      let swiperScroll = CommonFunc.getComponentRectOnScreen('list_offsetRemain');
      let top = swiperScroll.top;
      let right = swiperScroll.right;
      let left = swiperScroll.left;
      let bottom = swiperScroll.bottom;
      console.info("offsetRemain_0100 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom -top;
      let width = right - left;
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(left + width/2),Math.round(top + height*0.7),Math.round(left + width/2),Math.round(top + height*0.2));
      await CommonFunc.sleep(2000);
      
      let strJson = getInspectorByKey('offsetRemain');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[offsetRemain_0100] component obj is: " + JSON.stringify(obj));
      console.info("[offsetRemain_0100] content obj is: " + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content > 0).assertEqual(true);  
      console.info('[offsetRemain_0100] END');
      done();
    });

  })
}

