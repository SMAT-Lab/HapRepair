/*
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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function NavigationEnableDragBar() {
  describe('NavigationEnableDragBar', () => {
    beforeEach(async (done: Function) => {
      console.info("NavigationEnableDragBar beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/EnableDragBar/NavigationEnableDragBar',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationEnableDragBar state pages:" + JSON.stringify(pages));
        if (!("NavigationEnableDragBar" == pages.name)) {
          console.info("get NavigationEnableDragBar pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationEnableDragBar page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationEnableDragBar page error:" + err);
      }
      console.info("NavigationEnableDragBar beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavigationEnableDragBar after each called");
    });


    /**
     * @tc.number    NavigationEnableDragBar_0100
     * @tc.name      NavigationEnableDragBar_0100
     * @tc.desc      Test NavigationEnableDragBar
     * @tc.level     Level 2
     */
    it('NavigationEnableDragBar_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('NavigationEnableDragBar_0100 START');
      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      // The pc device does not have a DragBar.
      if (globalThis.winLeft == 0) {
        let btn1 = await driver.findComponent(ON.id('drag_true'));
        await btn1.click()
        console.info('[NavigationEnableDragBar_0100] click drag true');
        await CommonFunc.sleep(1000);
        await driver.assertComponentExist(ON.type('DragBarItem'));

        btn1 = await driver.findComponent(ON.id('drag_false'));
        await btn1.click()
        console.info('[NavigationEnableDragBar_0100] click drag false');
        await CommonFunc.sleep(1000);

        let comps = await driver.findComponent(ON.type('DragBarItem'));
        expect(comps).assertNull();
      }

      console.info('NavigationEnableDragBar_0100 END');
      done();
    });


  })
}