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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import router from '@ohos.router';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function Navigation1() {
  describe('Navigation1', () => {
    beforeEach(async (done: Function) => {
      console.info("appendChildTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/CustomNavigation/Navigation1',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get Navigation1 state pages: " + JSON.stringify(pages));
        if (!("Navigation1" == pages.name)) {
          console.info("get Navigation1 state pages.name: " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push Navigation1 success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Navigation1 page error: " + err);
        expect().assertFail();
      }
      done();
    });

    /**
     * @tc.number    Navigation1_0100
     * @tc.name      Navigation1_0100
     * @tc.desc      api
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('Navigation1_0100', Level.LEVEL2, async (done: Function) => {
      console.info('Navigation1_0100 START');
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      let button = await driver.findComponent(ON.id('removeIndexes'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('removeText');
      let obj: ESObject = JSON.parse(strJson);
      expect(obj.$attrs.content).assertEqual('0');
      console.info('[Navigation1_0100] END');
      done();
    });


    /**
     * @tc.number    Navigation1_0200
     * @tc.name      Navigation1_0200
     * @tc.desc      api
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('Navigation1_0200', Level.LEVEL2, async (done: Function) => {
      console.info('Navigation1_0200 START');
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      let button = await driver.findComponent(ON.id('deleteName'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('deleteText');
      let obj: ESObject = JSON.parse(strJson);
      expect(obj.$attrs.content).assertEqual('0');
      console.info('[Navigation1_0200] END');
      done();
    });


    /**
     * @tc.number    Navigation1_0300
     * @tc.name      Navigation1_0300
     * @tc.desc      api
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('Navigation1_0300', Level.LEVEL2, async (done: Function) => {
      console.info('Navigation1_0300 START');
      let strJson = getInspectorByKey('NavigationText');
      let obj: ESObject = JSON.parse(strJson);
      expect(obj.$attrs.content).assertEqual('pageOne');
      console.info('[Navigation1_0300] END');
      done();
    });
  })
}

