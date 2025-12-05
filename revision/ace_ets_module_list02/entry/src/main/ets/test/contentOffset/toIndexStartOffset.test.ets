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

export default function toIndexStartOffset() {
  describe('toIndexStartOffset', () => {
    beforeEach(async (done: Function) => {
      console.info("toIndexStartOffset beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/contentOffset/toIndexStartOffset",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get toIndexStartOffset state pages:" + JSON.stringify(pages));
        if (!("toIndexStartOffset" == pages.name)) {
          console.info("get toIndexStartOffset pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push toIndexStartOffset page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push toIndexStartOffset page error:" + err);
      }
      console.info("toIndexStartOffset beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("toIndexStartOffset after each called");
    });

    /**
     * @tc.number    toIndexStartOffset_001
     * @tc.name      toIndexStartOffset_0100
     * @tc.desc      The alignment method of scrollToIndex is AUTO.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('toIndexStartOffset_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[toIndexStartOffset_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('toIndex8_auto'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let list = CommonFunc.getComponentRectOnScreen('List_scrollToIndex8_List');
      let listItem = CommonFunc.getComponentRectOnScreen('List_scrollToIndex8_ListItem_8');
      console.info('[toIndexStartOffset_0100] list.bottom: ' + list.bottom);
      console.info('[toIndexStartOffset_0100] listItem.bottom: ' + listItem.bottom);
      await CommonFunc.sleep(1000);
      expect(Math.abs(Math.round(list.bottom - listItem.bottom) - Math.round(vp2px(10))) <=1 ).assertTrue();
      console.info('[toIndexStartOffset_0100] END');
      done();
    });
  })
}