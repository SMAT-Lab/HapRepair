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

export default function BarStylePage() {
  describe('BarStylePage', () => {
    beforeEach(async (done: Function) => {
      console.info('BarStylePage beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/BarStylePage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get BarStylePage state pages:' + JSON.stringify(pages));
        if (!('BarStylePage' == pages.name)) {
          console.info('get BarStylePage pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push BarStylePage page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push BarStylePage page error:' + err);
      }
      console.info('BarStylePage beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('BarStylePage after each called');
    });


    /**
     * @tc.number    BarStylePage_0100
     * @tc.name      BarStylePage_0100
     * @tc.desc      Test barStyle.STANDARD
     * @tc.level     Level 0
     */
    it('BarStylePage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('BarStylePage_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let component1: Component = await driver.findComponent(ON.type('TitleBar'));
      let titleRect = await component1.getBounds();
      let titleBottom = titleRect.bottom;
      console.info('BarStylePage_0100 titleBottom', titleBottom);

      let nav: Component = await driver.findComponent(ON.id('switchId'));
      let btnRect = await nav.getBounds();
      let btnTop = btnRect.top;
      console.info('BarStylePage_0100 btnTop', btnTop);
      expect(titleBottom == btnTop).assertTrue();

      let btn1 = await driver.findComponent(ON.id('switchId'));
      await btn1.click()
      console.info('[BarStylePage_0100] btn1 click');
      await CommonFunc.sleep(3000);

      let component2: Component = await driver.findComponent(ON.type('TitleBar'));
      let titleRect2 = await component2.getBounds();
      let titleBottom2 = titleRect2.bottom;
      console.info('BarStylePage_0100 titleBottom2', titleBottom2);

      let nav2: Component = await driver.findComponent(ON.id('switchId'));
      let btnRect2 = await nav2.getBounds();
      let btnTop2 = btnRect2.top;
      console.info('BarStylePage_0100 btnTop2', btnTop2);
      expect(titleBottom2 > btnTop2).assertTrue();

      console.info('BarStylePage_0100 END');
      done();
    });

  })
}