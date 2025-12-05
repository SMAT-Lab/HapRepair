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
import CommonFunc from '../../MainAbility/common/Common';
import {UiComponent, UiDriver, Component, Driver, MouseButton, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function ListAddApi() {
  describe('ListAddApi',  () => {
    beforeEach(async (done: Function) =>  {
      console.info("ListAddApi beforeEach start");
      try {
        router.clear();
        let pages = router.getState();
        console.info("get ListAddApi state pages:" + JSON.stringify(pages));
        if (!("ListAddApi" == pages.name)) {
          console.info("get  Context pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl({url:"MainAbility/pages/List/ListAddApi"})
          await CommonFunc.sleep(2000);
          console.info("push ListAddApi page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push ListAddApi page error:" + err);
      }
      console.info("ListAddApi foreEach end");
      done();
    });

    /*
     * @tc.number     : SUB_ACE_UI_COMPONENT_List_0010
     * @tc.name       : ListAddApi_001
     * @tc.desc       : RichEditor Test
     * @tc.size       : MediumTest
     * @tc.type       : Function
     * @tc.level      : Level 3
     */
    it('ListAddApi_001', Level.LEVEL3, async (done: Function) => {
      console.info('[ListAddApi_001] START');
      await CommonFunc.sleep(1000);
      let driver: Driver = Driver.create();
      let button: Component = await driver.findComponent(ON.id('ListAddApiBt'));
      await button.click()
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('ListAddApiText1');
      await CommonFunc.sleep(200);
      console.info("[ListAddApi_001] component obj is: " + JSON.stringify(JSON.parse(strJson)));
      await CommonFunc.sleep(1000);

      expect(JSON.parse(strJson).$attrs.content).assertEqual('true');
      console.info('[ListAddApi_001] END');
      done();
    });

  })
}
