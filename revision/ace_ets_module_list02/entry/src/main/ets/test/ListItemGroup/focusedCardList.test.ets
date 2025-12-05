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

export default function focusedCardList() {
  describe('focusedCardList', () => {
    beforeEach(async (done: Function) => {
      console.info("focusedCardList beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/focusedCardList",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get focusedCardList state pages:" + JSON.stringify(pages));
        if (!("focusedCardList" == pages.name)) {
          console.info("get focusedCardList pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push focusedCardList page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push focusedCardList page error:" + err);
      }
      console.info("focusedCardList beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("focusedCardList after each called");
    });

    /**
     * @tc.number    focusedCardList_001
     * @tc.name      focusedCardList_0100
     * @tc.desc      Test the card style of the List component in focus state.
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('focusedCardList_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[focusedCardList_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      await driver.triggerKey(2049);
      await CommonFunc.sleep(1000);

      let btn: Component = await driver.findComponent(ON.id('listItem_focused_card'));
      let btn_isFocusedValue = await btn.isFocused()
      expect(btn_isFocusedValue).assertEqual(true);
      console.info('[focusedCardList_0100] ListItem isFocused');

      console.info('[focusedCardList_0100] END');
      done();
    });
  })
}
