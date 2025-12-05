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
import { Driver, Component, UiWindow, ON } from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function offsetAlignStart() {
  describe('offsetAlignStart', () => {
    beforeEach(async (done: Function) => {
      console.info("offsetAlignStart beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/contentOffset/offsetAlignStart",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get offsetAlignStart state pages:" + JSON.stringify(pages));
        if (!("offsetAlignStart" == pages.name)) {
          console.info("get offsetAlignStart pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push offsetAlignStart page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push offsetAlignStart page error:" + err);
      }
      console.info("offsetAlignStart beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("offsetAlignStart after each called");
    });

    /**
     * @tc.number    offsetAlignStart_001
     * @tc.name      offsetAlignStart_0100
     * @tc.desc      Limit scrolling, ScrollSnapAlign.START Testing.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('offsetAlignStart_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[offsetAlignStart_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('scroll_align_offset_top'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let list = CommonFunc.getComponentRectOnScreen('align_startOffset_start');
      let listItem = CommonFunc.getComponentRectOnScreen('align_startOffset_start_0');
      console.info('[offsetAlignStart_0100] list.top: ' + list.top);
      console.info('[offsetAlignStart_0100] listItem.top: ' + listItem.top);
      await CommonFunc.sleep(1000);
      expect(Math.abs(Math.round(listItem.top - list.top) - Math.round(vp2px(10))) <= 1 ).assertTrue();
      console.info('[offsetAlignStart_0100] END');
      done();
    });
  })
}
