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
import {UiComponent, Driver, Component, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function listDidJsunit() {
  describe('listDidJsunit', () => {
    beforeEach(async (done: Function) => {
      console.info("listDidJsunit beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/listDid",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listDidJsunit state pages:" + JSON.stringify(pages));
        if (!("listDid" == pages.name)) {
          console.info("get listDidJsunit pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push listDidJsunit page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listDidJsunit page error:" + err);
      }
      console.info("listDidJsunit beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listDidJsunit after each called");
    });

    /**
     * @tc.number    List_ONDIDSCROLL_0100
     * @tc.name      listDidJsunit_0100
     * @tc.desc      The list component triggers the onDidScroll event
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('listDidJsunit_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[listDidJsunit_0100] START');
      let listRect = CommonFunc.getComponentRectOnScreen('scrollSource_list')
      let driver = Driver.create()
      await driver.fling({
        x: Number.parseInt(`${(listRect.right - listRect.left) * 0.5 + listRect.left}`),
        y: Number.parseInt(`${(listRect.bottom - listRect.top) * 0.6 + listRect.top}`)
      },
        {
          x: Number.parseInt(`${(listRect.right - listRect.left) * 0.5 + listRect.left}`),
          y: Number.parseInt(`${(listRect.bottom - listRect.top) * 0.2 + listRect.top}`)
        }, 5, 1000);
      await CommonFunc.sleep(2000);

      let strJson = getInspectorByKey('list_onDidScroll');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[listDidJsunit_0100] component obj is: " + JSON.stringify(obj));
      console.info("[listDidJsunit_0100] content: " + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('scrollState: 0');
      console.info('[listDidJsunit_0100] END');
      done();
    });
  })
}
