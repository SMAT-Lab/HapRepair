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

export default function offsetAlignEnd() {
  describe('offsetAlignEnd', () => {
    beforeEach(async (done: Function) => {
      console.info("offsetAlignEnd beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/contentOffset/offsetAlignEnd",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get offsetAlignEnd state pages:" + JSON.stringify(pages));
        if (!("offsetAlignEnd" == pages.name)) {
          console.info("get offsetAlignEnd pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push offsetAlignEnd page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push offsetAlignEnd page error:" + err);
      }
      console.info("offsetAlignEnd beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("offsetAlignEnd after each called");
    });

    /**
     * @tc.number    offsetAlignEnd_001
     * @tc.name      offsetAlignEnd_0100
     * @tc.desc      Limit scrolling, ScrollSnapAlign.END Testing.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('offsetAlignEnd_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[offsetAlignEnd_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('scroll_align_offset_btm'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let list = CommonFunc.getComponentRectOnScreen('align_endOffset_end');
      let listItem = CommonFunc.getComponentRectOnScreen('align_endOffset_end_9');
      console.info('[offsetAlignEnd_0100] list.bottom: ' + list.bottom);
      console.info('[offsetAlignEnd_0100] listItem.bottom: ' + listItem.bottom);
      await CommonFunc.sleep(1000);
      expect(Math.abs(Math.round(list.bottom - listItem.bottom) - Math.round(vp2px(10))) <= 1 ).assertTrue();
      console.info('[offsetAlignEnd_0100] END');
      done();
    });
  })
}
