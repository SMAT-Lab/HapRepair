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

export default function singleHozDiff() {
  describe('singleHozDiff', () => {
    beforeEach(async (done: Function) => {
      console.info("singleHozDiff beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/singleHozDiff",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get singleHozDiff state pages:" + JSON.stringify(pages));
        if (!("singleHozDiff" == pages.name)) {
          console.info("get singleHozDiff pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push singleHozDiff page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push singleHozDiff page error:" + err);
      }
      console.info("singleHozDiff beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("singleHozDiff after each called");
    });

    /**
     * @tc.number    singleHozDiff_001
     * @tc.name      singleHozDiff_0100
     * @tc.desc      List component with different heights arranged horizontally in a single column, jumps to the specified index.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('singleHozDiff_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[singleHozDiff_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('scroller_group_hoz_diff'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let list = CommonFunc.getComponentRectOnScreen('group_hoz_diff_list');
      let listItem = CommonFunc.getComponentRectOnScreen('group_hoz_diff_item');
      await CommonFunc.sleep(1000);
      expect(Math.abs(listItem.top - list.top) <= 1).assertEqual(true);
      console.info('[singleHozDiff_0100] END');
      done();
    });
  })
}