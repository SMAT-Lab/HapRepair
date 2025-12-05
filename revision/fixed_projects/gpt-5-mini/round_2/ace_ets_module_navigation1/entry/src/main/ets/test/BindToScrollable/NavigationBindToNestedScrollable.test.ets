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

export default function NavigationBindToNestedScrollable() {
  describe('NavigationBindToNestedScrollable', () => {
    beforeEach(async (done: Function) => {
      console.info("NavigationBindToNestedScrollable beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/BindToScrollable/NavigationBindToNestedScrollable',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationBindToNestedScrollable state pages:" + JSON.stringify(pages));
        if (!("NavigationBindToNestedScrollable" == pages.name)) {
          console.info("get NavigationBindToNestedScrollable pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationBindToNestedScrollable page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationBindToNestedScrollable page error:" + err);
      }
      console.info("NavigationBindToNestedScrollable beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavigationBindToNestedScrollable after each called");
    });


    /**
     * @tc.number    NavigationBindToNestedScrollable_0100
     * @tc.name      NavigationBindToNestedScrollable_0100
     * @tc.desc      Test NavigationBindToNestedScrollable
     * @tc.level     Level 2
     */
    it('NavigationBindToNestedScrollable_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('NavigationBindToNestedScrollable_0100 START');
      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.text('push pageOne'));
      await btn1.click()
      console.info('[NavigationBindToNestedScrollable_0100] click push pageOne');
      await CommonFunc.sleep(1000);

      let navigation: Component = await driver.findComponent(ON.type('Navigation'));
      let rect2 = await navigation.getBounds();
      let navigationWidth = rect2.right;
      console.info('[NavigationBindToNestedScrollable_0100] navigationWidth is ', navigationWidth);

      await driver.assertComponentExist(ON.type('TitleBar'));
      if (navigationWidth < 1500) {
        await driver.assertComponentExist(ON.type('ToolBar'));
        console.info('[NavigationBindToNestedScrollable_0100] ToolBar is exist');
        await driver.delayMs(1000);
      }

      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let scrollBarRect = await scrollBar.getBounds();
      let top = scrollBarRect.top;
      let right = scrollBarRect.right;
      let bottom = scrollBarRect.bottom;
      
      await driver.swipe(Math.round(right - vp2px(11)), Math.round(top + vp2px(12)), Math.round(right - vp2px(11)), Math.round(bottom));
      await driver.delayMs(500)

      await driver.assertComponentExist(ON.type('TitleBar'));

      let tooBarTop1 = 0;
      let tooBarBottom1 = 0;
      if (navigationWidth < 1500) {
        await driver.assertComponentExist(ON.type('ToolBar'));
        await driver.delayMs(1000);
        let component3: Component  = await driver.findComponent(ON.type('ToolBar'));
        let toolBarRect = await component3.getBounds();
        let tooBarTop1 = toolBarRect.top;
        let tooBarBottom1 = toolBarRect.bottom;
        console.info('NavigationBindToNestedScrollable_0100 test tooBarTop1: ', tooBarTop1);
        console.info('NavigationBindToNestedScrollable_0100 test tooBarBottom1: ', tooBarBottom1);
      }

      await driver.swipe(Math.round(right - vp2px(11)), Math.round(bottom - vp2px(12)), Math.round(right - vp2px(11)), Math.round(top));
      await driver.delayMs(500)
      let list: Component = await driver.findComponent(ON.type('List'));
      
      let listRect = await list.getBounds();
      let listTop = listRect.top;
      let listRight = listRect.right;
      let listBottom = listRect.bottom;

      await driver.swipe(Math.round(listRight - vp2px(11)), Math.round(listTop + vp2px(12)), Math.round(listRight - vp2px(11)), Math.round(listBottom));
      await driver.delayMs(500)

      let component1: Component = await driver.findComponent(ON.type('TitleBar').text('pageOne'));
      let point = await component1.getBoundsCenter();
      let x = point.x;
      let y = point.y;
      console.info('NavigationBindToNestedScrollable_0100 x', x);
      console.info('NavigationBindToNestedScrollable_0100 y', y);

      let nav: Component = await driver.findComponent(ON.id('navId'));
      let rect = await nav.getBounds();
      let windowTop = rect.top;
      console.info('NavigationBindToNestedScrollable_0100 windowTop', windowTop);
      expect(y < windowTop).assertTrue();
     
      let component2: Component  = await driver.findComponent(ON.type('ToolBar'));
      if (component2) {
        let toolBarRect2 = await component2.getBounds();
        let tooBarTop2 = toolBarRect2.top;
        let tooBarBottom2 = toolBarRect2.bottom;
        console.info('NavigationBindToNestedScrollable_0100 tooBarTop2: ', tooBarTop2);
        console.info('NavigationBindToNestedScrollable_0100 tooBarBottom2: ', tooBarBottom2);
        expect(Math.abs(Math.round(tooBarTop2 - tooBarTop1) - Math.round(tooBarBottom1 - tooBarTop1)) <= 1 ).assertTrue();
      } else {
        expect(component2).assertEqual(null); 
      }

      console.info('NavigationBindToNestedScrollable_0100 END');
      done();
    });
  })
}