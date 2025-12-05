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

export default function focusedList() {
  describe('focusedList', () => {
    beforeEach(async (done: Function) => {
      console.info("focusedList beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/focusedList",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get focusedList state pages:" + JSON.stringify(pages));
        if (!("focusedList" == pages.name)) {
          console.info("get focusedList pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push focusedList page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push focusedList page error:" + err);
      }
      console.info("focusedList beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("focusedList after each called");
    });

    /**
     * @tc.number    focusedList_001
     * @tc.name      focusedList_0100
     * @tc.desc      List Component Focus Test.
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('focusedList_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[focusedList_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      await driver.triggerKey(2049);
      await CommonFunc.sleep(2000);

      let strJson = getInspectorByKey('focused_color_1');
      console.info("[scrollAniFalse_0100] component obj is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[scrollAniFalse_0100] backgroundColor: " + JSON.stringify(JSON.parse(strJson).$attrs.backgroundColor));
      await CommonFunc.sleep(1000);
      expect(JSON.parse(strJson).$attrs.backgroundColor).assertEqual('#FFFF0000');
      console.info('[focusedList_0100] item-1 isFocused');

      console.info('[focusedList_0100] END');
      done();
    });
  })
}
