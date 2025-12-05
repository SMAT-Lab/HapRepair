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

export default function lanesHeaderGroup() {
  describe('lanesHeaderGroup', () => {
    beforeEach(async (done: Function) => {
      console.info("lanesHeaderGroup beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/lanesHeaderGroup",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get lanesHeaderGroup state pages:" + JSON.stringify(pages));
        if (!("lanesHeaderGroup" == pages.name)) {
          console.info("get lanesHeaderGroup pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push lanesHeaderGroup page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push lanesHeaderGroup page error:" + err);
      }
      console.info("lanesHeaderGroup beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("lanesHeaderGroup after each called");
    });

    /**
     * @tc.number    lanesHeaderGroup_001
     * @tc.name      lanesHeaderGroup_0100
     * @tc.desc      List component with top and bottom suction vertically arranged in multiple columns, jumps to the specified index.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('lanesHeaderGroup_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[lanesHeaderGroup_0100] START');
      await CommonFunc.sleep(1000);
      let groupHeader = CommonFunc.getComponentRectOnScreen('group_ver_list_head');
      let listItem = CommonFunc.getComponentRectOnScreen('group_ver_item_head');
      console.info('[lanesHeaderGroup_0100] groupHeader.bottom: ' + groupHeader.bottom);
      console.info('[lanesHeaderGroup_0100] listItem.top: ' + listItem.top);
      expect(Math.abs(listItem.top - groupHeader.bottom) <= 1).assertTrue();
      let driver = Driver.create();
      await CommonFunc.sleep(1000);
      let btn: Component = await driver.findComponent(ON.id('itemInGroup_ver_header'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let text01 = await driver.findComponent(ON.text("Group:2, Item:3"))
      let textContent = await text01.getText();
      expect(textContent).assertEqual("Group:2, Item:3");
      console.info('[lanesHeaderGroup_0100] END');
      done();
    });
  })
}