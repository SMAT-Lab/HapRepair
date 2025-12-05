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

export default function paddingStartOffset() {
  describe('paddingStartOffset', () => {
    beforeEach(async (done: Function) => {
      console.info("paddingStartOffset beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/contentOffset/paddingStartOffset",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get paddingStartOffset state pages:" + JSON.stringify(pages));
        if (!("paddingStartOffset" == pages.name)) {
          console.info("get paddingStartOffset pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push paddingStartOffset page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push paddingStartOffset page error:" + err);
      }
      console.info("paddingStartOffset beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("paddingStartOffset after each called");
    });

    /**
     * @tc.number    paddingStartOffset_001
     * @tc.name      paddingStartOffset_0100
     * @tc.desc      Set padding and contentStartOffset for the list component.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('paddingStartOffset_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[paddingStartOffset_0100] START');
      await CommonFunc.sleep(1000);
      let list = CommonFunc.getComponentRectOnScreen('list_padStart');
      let listItem = CommonFunc.getComponentRectOnScreen('listItem_padStart_1');
      console.info('[paddingStartOffset_0100] list.top: ' + list.top);
      console.info('[paddingStartOffset_0100] listItem.top: ' + listItem.top);
      await CommonFunc.sleep(1000);
      expect(Math.abs(Math.round(listItem.top - list.top) - Math.round(vp2px(10))) <= 1 ).assertTrue();
      console.info('[paddingStartOffset_0100] END');
      done();
    });
  })
}