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
export default function getTest() {
  describe('getTest', () => {
    beforeEach(async (done: Function) => {
      console.info("getTest beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/childrenMainSize/get",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get getTest state pages: " + JSON.stringify(pages));
        if (!("get" == pages.name)) {
          console.info("get getTest state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push getTest success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push getTest page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /**
     * @tc.number    getTest_0100
     * @tc.name      getTest_0100
     * @tc.desc      api
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('getTest_0100', Level.LEVEL2, async (done: Function) => {
      console.info('getTest_0100 START');
      await CommonFunc.sleep(1000);
      let driver: Driver = Driver.create();
      let get_childDefaultSize: Component = await driver.findComponent(ON.id('get_childDefaultSize'));
      await get_childDefaultSize.click();
      await CommonFunc.sleep(1000);
      let get = getInspectorByKey('get');
      console.info("[getTest_0100] attrs is: " + JSON.stringify(JSON.parse(get).$attrs));
      expect(JSON.parse(get).$attrs.content).assertEqual('get childDefaultSize:0');
      console.info('[getTest_0100] END');
      done();
    });
  })
}