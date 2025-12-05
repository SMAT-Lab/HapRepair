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

export default function NavigationBindToScrollable() {
  describe('NavigationBindToScrollable', () => {
    beforeEach(async (done: Function) => {
      console.info("NavigationBindToScrollable beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/BindToScrollable/NavigationBindToScrollable',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationBindToScrollable state pages:" + JSON.stringify(pages));
        if (!("NavigationBindToScrollable" == pages.name)) {
          console.info("get NavigationBindToScrollable pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationBindToScrollable page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationBindToScrollable page error:" + err);
      }
      console.info("NavigationBindToScrollable beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavigationBindToScrollable after each called");
    });


    /**
     * @tc.number    NavigationBindToScrollable_0100
     * @tc.name      NavigationBindToScrollable_0100
     * @tc.desc      Test NavigationBindToScrollable
     * @tc.level     Level 2
     */
    it('NavigationBindToScrollable_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('NavigationBindToScrollable_0100 START');
      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.text('push pageOne'));
      await btn1.click()
      console.info('[NavigationBindToScrollable_0100] click push pageOne');
      await CommonFunc.sleep(1000);

      let navigation: Component = await driver.findComponent(ON.type('Navigation'));
      let rect = await navigation.getBounds();
      let navigationWidth = rect.right;
      console.info('[NavigationBindToScrollable_0100] navigationWidth is ', navigationWidth);

      await driver.assertComponentExist(ON.type('TitleBar'));
      if (navigationWidth < 1500) {
        await driver.assertComponentExist(ON.type('ToolBar'));
        console.info('[NavigationBindToScrollable_0100] ToolBar is exist');
        await driver.delayMs(1000);
      }

      let scrollBar: Component = await driver.findComponent(ON.type('List'));
      await scrollBar.scrollToBottom();
      await CommonFunc.sleep(5000);

      await driver.assertComponentExist(ON.type('TitleBar'));
      if (navigationWidth < 1500) {
        await driver.assertComponentExist(ON.type('ToolBar'));
        console.info('[NavigationBindToScrollable_0100] ToolBar is exist');
        await driver.delayMs(1000);
      }

      let titleBar: Component | null = await driver.findComponent(ON.type('TitleBar'));
      expect(titleBar !== null).assertTrue();

      console.info('NavigationBindToScrollable_0100 END');
      done();
    });


  })
}