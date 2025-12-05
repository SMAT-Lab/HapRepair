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

export default function HideToolBarNavPage() {
  describe('HideToolBarNavPage', () => {
    beforeEach(async (done: Function) => {
      console.info('HideToolBarNavPage beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/HideToolBarNavPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get HideToolBarNavPage state pages:' + JSON.stringify(pages));
        if (!('HideToolBarNavPage' == pages.name)) {
          console.info('get HideToolBarNavPage pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push HideToolBarNavPage page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push HideToolBarNavPage page error:' + err);
      }
      console.info('HideToolBarNavPage beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('HideToolBarNavPage after each called');
    });


    /**
     * @tc.number    HideToolBarNavPage_0100
     * @tc.name      HideToolBarNavPage_0100
     * @tc.desc      Test NavDestination hideToolBar
     * @tc.level     Level 0
     */
    it('HideToolBarNavPage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('HideToolBarNavPage_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let pushBtn = await driver.findComponent(ON.id('pushId'));
      await pushBtn.click()
      console.info('[HideToolBarNavPage_0100] pushBtn click');
      await CommonFunc.sleep(1000);

      await driver.assertComponentExist(ON.type('ToolBar'));

      let switchBtn = await driver.findComponent(ON.id('switchId'));
      await switchBtn.click()
      console.info('[HideToolBarNavPage_0100] switchBtn click');
      await CommonFunc.sleep(1000);

      let component: Component  = await driver.findComponent(ON.type('ToolBar'));
      expect(component).assertEqual(null);

      console.info('HideToolBarNavPage_0100 END');
      done();
    });

  })
}