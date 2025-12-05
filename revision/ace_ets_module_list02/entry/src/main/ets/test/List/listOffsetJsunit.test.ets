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

export default function listOffsetJsunit() {
  describe('listOffsetJsunit', () => {
    beforeEach(async (done: Function) => {
      console.info("listOffsetJsunit beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/listOffset",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listOffsetJsunit state pages:" + JSON.stringify(pages));
        if (!("listOffset" == pages.name)) {
          console.info("get listOffsetJsunit pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push listOffsetJsunit page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listOffsetJsunit page error:" + err);
      }
      console.info("listOffsetJsunit beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listOffsetJsunit after each called");
    });

    /**
     * @tc.number    listOffsetJsunit_001
     * @tc.name      listOffsetJsunit_001
     * @tc.desc      Slide according to the specified offset when returning ScrollResult.
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('listOffsetJsunit_001', Level.LEVEL2, async (done: Function) => {
      console.info('[listOffsetJsunit_001] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let swiperScroll = CommonFunc.getComponentRectOnScreen('offset_list');
      let top = swiperScroll.top;
      let right = swiperScroll.right;
      let left = swiperScroll.left;
      let bottom = swiperScroll.bottom;
      console.info("listOffsetJsunit_001 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom -top;
      let width = right - left;
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(left + width/2),Math.round(top + height*0.7),Math.round(left + width/2),Math.round(top + height*0.2));
      await CommonFunc.sleep(2000);
      
      let strJson = getInspectorByKey('list_offset');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[listOffsetJsunit_001] component obj is: " + JSON.stringify(obj));
      expect(obj.$attrs.content).assertEqual('ScrollResult: 10');
      console.info('[listOffsetJsunit_001] END');
      done();
    });
  })
}
