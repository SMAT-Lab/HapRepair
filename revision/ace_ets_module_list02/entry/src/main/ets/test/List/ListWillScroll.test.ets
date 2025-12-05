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
import {UiComponent, Driver, Component, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function ListWillScroll() {
  describe('ListWillScroll', () => {
    beforeEach(async (done: Function) => {
      console.info("ListWillScroll beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/ListWillScroll',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get ListWillScroll state pages: " + JSON.stringify(pages));
        if (!("ListWillScroll" == pages.name)) {
          console.info("get ListWillScroll state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push ListWillScroll success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push ListWillScroll page error: " + err);
        expect().assertFail();
      }
      done();
    });
    /**
     * @tc.number    ListWillScroll_0100
     * @tc.name      ListWillScroll_0100
     * @tc.desc      api
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('ListWillScroll_0100', Level.LEVEL2, async (done: Function) => {
      console.info('ListWillScroll_0100 START');
      let swiperAnimation = CommonFunc.getComponentRectOnScreen('listScroll_4');
      let top = swiperAnimation.top;
      let right = swiperAnimation.right;
      let left = swiperAnimation.left;
      let bottom = swiperAnimation.bottom;
      console.info("ListWillScroll_0100 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom -top;
      let width = right - left;
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(left + width*0.5),Math.round(bottom - height/4),Math.round(left + width*0.5),Math.round(bottom - height/2));
      await CommonFunc.sleep(3000);
      let strJson = getInspectorByKey('listScroll_1');
      let strJson1 = getInspectorByKey('listScroll_2');
      let strJson2 = getInspectorByKey('listScroll_3');
      let obj: ESObject = JSON.parse(strJson);
      let obj1: ESObject = JSON.parse(strJson1);
      let obj2: ESObject = JSON.parse(strJson2);
      console.info("[ListWillScroll_0100] component obj is: " + JSON.stringify(obj));
      console.info("[ListWillScroll_0100] component obj1 is: " + JSON.stringify(obj1));
      console.info("[ListWillScroll_0100] component obj2 is: " + JSON.stringify(obj2));
      expect(obj.$attrs.content).assertEqual('onWillScroll');
      expect(obj1.$attrs.content).assertEqual('onDidScroll');
      expect(obj2.$attrs.content).assertEqual('onScrollVisibleContentChange');
      console.info('[ListWillScroll_0100] END');
      done();
    });
  })
}

