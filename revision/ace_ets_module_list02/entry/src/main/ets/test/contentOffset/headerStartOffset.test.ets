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

export default function headerStartOffset() {
  describe('headerStartOffset', () => {
    beforeEach(async (done: Function) => {
      console.info("headerStartOffset beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/contentOffset/headerStartOffset",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get headerStartOffset state pages:" + JSON.stringify(pages));
        if (!("headerStartOffset" == pages.name)) {
          console.info("get headerStartOffset pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push headerStartOffset page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push headerStartOffset page error:" + err);
      }
      console.info("headerStartOffset beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("headerStartOffset after each called");
    });

    /**
     * @tc.number    headerStartOffset_001
     * @tc.name      headerStartOffset_0100
     * @tc.desc      ScrollToItemInGroup alignment method.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('headerStartOffset_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[headerStartOffset_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('startOffset_itemInGroup'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let groupHeader = CommonFunc.getComponentRectOnScreen('group_head_startOffset_2');
      let listItem = CommonFunc.getComponentRectOnScreen('group_item_startOffset_2_3');
      let list = CommonFunc.getComponentRectOnScreen('group_list_startOffset');
      console.info('[headerStartOffset_0100] groupHeader.top: ' + groupHeader.top);
      console.info('[headerStartOffset_0100] groupHeader.bottom: ' + groupHeader.bottom);
      console.info('[headerStartOffset_0100] listItem.top: ' + listItem.top);
      console.info('[headerStartOffset_0100] list.top: ' + list.top);
      await CommonFunc.sleep(1000);
      expect(Math.abs(listItem.top - groupHeader.bottom) <= 1).assertTrue();
      expect(Math.abs(Math.round(groupHeader.top - list.top) - Math.round(vp2px(10))) <= 1 ).assertTrue();
      console.info('[headerStartOffset_0100] END');
      done();
    });
  })
}