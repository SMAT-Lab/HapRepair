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

export default function listVisibleContent() {
  describe('listVisibleContent', () => {
    beforeEach(async (done: Function) => {
      console.info("listVisibleContent beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/listVisibleContent",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listVisibleContent state pages:" + JSON.stringify(pages));
        if (!("listVisibleContent" == pages.name)) {
          console.info("get listVisibleContent pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push listVisibleContent page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listVisibleContent page error:" + err);
      }
      console.info("listVisibleContent beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listVisibleContent after each called");
    });

    /* *
    * @tc.number: SUB_ACE_LIST_listVisibleContent_0100
    * @tc.name  : listVisibleContent_0100
    * @tc.desc  : Set the property value of mainVisibleContentPosition to false.
    * @tc.level : Level 2
    * @tc.type :
    * @tc.size :
    */
    it('listVisibleContent_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[listVisibleContent_0100] START');
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('list1_visible');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[listVisibleContent_0100] component obj is: " + JSON.stringify(obj));
      console.info("[listVisibleContent_0100] maintainVisibleContentPosition: " + JSON.stringify(obj.$attrs.maintainVisibleContentPosition));
      expect(obj.$attrs.maintainVisibleContentPosition).assertEqual(false);
      console.info('[listVisibleContent_0100] END');
      done();
    });

    /* *
    * @tc.number: SUB_ACE_LIST_listVisibleContent_0200
    * @tc.name  : listVisibleContent_0200
    * @tc.desc  : When inserting data above the display area, the visible content position will change accordingly.
    * @tc.level : Level 2
    * @tc.type :
    * @tc.size :
    */
    it('listVisibleContent_0200', Level.LEVEL2, async (done: Function) => {
      console.info('[listVisibleContent_0200] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('scrollTo_y'));
      await btn.click();
      await CommonFunc.sleep(3000);
      let btn2: Component = await driver.findComponent(ON.id('deleteItem_index'));
      await btn2.click();
      await CommonFunc.sleep(2000);
      let strJson = getInspectorByKey('list_num');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[listVisibleContent_0200] component obj is: " + JSON.stringify(obj));
      console.info("[listVisibleContent_0200] length: " + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('length: 49');
      console.info('[listVisibleContent_0200] END');
      done();
    });
  })
}
