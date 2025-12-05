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

export default function singleHozList() {
  describe('singleHozList', () => {
    beforeEach(async (done: Function) => {
      console.info("singleHozList beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/singleHozList",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get singleHozList state pages:" + JSON.stringify(pages));
        if (!("singleHozList" == pages.name)) {
          console.info("get singleHozList pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push singleHozList page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push singleHozList page error:" + err);
      }
      console.info("singleHozList beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("singleHozList after each called");
    });

    /**
     * @tc.number    singleHozList_001
     * @tc.name      singleHozList_0100
     * @tc.desc      List component arranged horizontally in a single column, jumps to the specified index.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('singleHozList_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[singleHozList_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('scroller_group_hoz'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let list = CommonFunc.getComponentRectOnScreen('group_hoz_list');
      let listItem = CommonFunc.getComponentRectOnScreen('group_hoz_item');
      await CommonFunc.sleep(1000);
      expect(Math.abs(listItem.top - list.top) <= 1).assertEqual(true);
      console.info('[singleHozList_0100] END');
      done();
    });
  })
}