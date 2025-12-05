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

export default function listScrollTo() {
  describe('listScrollTo', () => {
    beforeEach(async (done: Function) => {
      console.info("listScrollTo beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/listScrollTo",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listScrollTo state pages:" + JSON.stringify(pages));
        if (!("listScrollTo" == pages.name)) {
          console.info("get listScrollTo pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push listScrollTo page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listScrollTo page error:" + err);
      }
      console.info("listScrollTo beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listScrollTo after each called");
    });

    /**
     * @tc.number    listScrollTo_001
     * @tc.name      listScrollTo_0100
     * @tc.desc      List component binding scroller event sliding 100vp.
     * @tc.level     Level 3
     * @tc.type
     * @tc.size
     */
    it('listScrollTo_0100', Level.LEVEL3, async (done: Function) => {
      console.info('[listScrollTo_0100] START');
      await CommonFunc.sleep(1000);
      let listItem1 = CommonFunc.getComponentRectOnScreen('list_scroll_100_2');
      console.info('[listScrollTo_0100] listItem1.bottom: ' + listItem1.bottom);
      await CommonFunc.sleep(2000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('list_scrollTo_1'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let listItem2 = CommonFunc.getComponentRectOnScreen('list_scroll_100_2');
      console.info('[listScrollTo_0100] listItem2.bottom: ' + listItem2.bottom);
      await CommonFunc.sleep(1000);
      expect(Math.abs(Math.round(listItem1.bottom - listItem2.bottom) - Math.round(vp2px(100))) <=1 ).assertTrue();
      console.info('[listScrollTo_0100] END');
      done();
    });
  })
}