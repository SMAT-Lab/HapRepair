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

export default function offsetEdgeBtm() {
  describe('offsetEdgeBtm', () => {
    beforeEach(async (done: Function) => {
      console.info("offsetEdgeBtm beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/contentOffset/offsetEdgeBtm",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get offsetEdgeBtm state pages:" + JSON.stringify(pages));
        if (!("offsetEdgeBtm" == pages.name)) {
          console.info("get offsetEdgeBtm pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push offsetEdgeBtm page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push offsetEdgeBtm page error:" + err);
      }
      console.info("offsetEdgeBtm beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("offsetEdgeBtm after each called");
    });

    /**
     * @tc.number    offsetEdgeBtm_001
     * @tc.name      offsetEdgeBtm_001
     * @tc.desc      Controller scrollEdge scrolls to the bottom of the list
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('offsetEdgeBtm_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[offsetEdgeBtm_0100] START');
      let swiperList = CommonFunc.getComponentRectOnScreen('edge_endOffset_btm');
      let top = swiperList.top;
      let right = swiperList.right;
      let left = swiperList.left;
      let bottom = swiperList.bottom;
      console.info("offsetEdgeBtm value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom -top;
      let width = right - left;
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(left + width/2),Math.round(top + height*0.8),Math.round(left + width/2),Math.round(top + height*0.2));
      await CommonFunc.sleep(3000);

      let btn: Component = await driver.findComponent(ON.id('offset_scrollEdge_btm'));
      await btn.click();
      await CommonFunc.sleep(2000);

      let list = CommonFunc.getComponentRectOnScreen('edge_endOffset_btm');
      let listItem = CommonFunc.getComponentRectOnScreen('edge_endOffset_btm_15');
      console.info('[offsetEdgeBtm_0100] list.bottom: ' + list.bottom);
      console.info('[offsetEdgeBtm_0100] listItem.bottom: ' + listItem.bottom);
      await CommonFunc.sleep(1000);
      expect(Math.abs(Math.round(list.bottom - listItem.bottom) - Math.round(vp2px(10))) <= 1 ).assertTrue();
      console.info('[offsetEdgeBtm_0100] END');
      done();
    });
  })
}
