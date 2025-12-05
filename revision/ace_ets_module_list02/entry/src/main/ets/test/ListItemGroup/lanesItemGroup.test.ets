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

export default function lanesItemGroup() {
  describe('lanesItemGroup', () => {
    beforeEach(async (done: Function) => {
      console.info("lanesItemGroup beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/lanesItemGroup",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get lanesItemGroup state pages:" + JSON.stringify(pages));
        if (!("lanesItemGroup" == pages.name)) {
          console.info("get lanesItemGroup pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push lanesItemGroup page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push lanesItemGroup page error:" + err);
      }
      console.info("lanesItemGroup beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("lanesItemGroup after each called");
    });

    /**
     * @tc.number    lanesItemGroup_001
     * @tc.name      lanesItemGroup_0100
     * @tc.desc      List component vertically arranged in multiple columns, jumps to the specified index.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('lanesItemGroup_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[lanesItemGroup_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('itemInGroup_ver_item'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let list = CommonFunc.getComponentRectOnScreen('group_ver_list');
      let listItem = CommonFunc.getComponentRectOnScreen('group_ver_2_3');
      await CommonFunc.sleep(1000);
      console.info('[lanesItemGroup_0100] listItem.top' + listItem.top);
      console.info('[lanesItemGroup_0100] list.top' + list.top);
      expect(Math.abs(listItem.top - list.top) <= 1).assertEqual(true);
      console.info('[lanesItemGroup_0100] END');
      done();
    });
  })
}