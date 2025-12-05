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

import CommonFunc from '../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, Level, expect } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import events_emitter from '@ohos.events.emitter'

export default function NavDestinationStateActive() {
  describe('NavDestinationStateActive', () => {
    beforeEach(async (done: Function) => {
      console.info("NavDestinationStateActive beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/NavDestinationStateActive',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavDestinationStateActive state pages:" + JSON.stringify(pages));
        if (!("NavDestinationStateActive" == pages.name)) {
          console.info("get NavDestinationStateActive pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavDestinationStateActive page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavDestinationStateActive page error:" + err);
      }
      console.info("NavDestinationStateActive beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavDestinationStateActive after each called");
    });


    /**
     * @tc.number    NavDestinationStateActive_0100
     * @tc.name      NavDestinationStateActive_0100
     * @tc.desc      Test NavDestinationState ON_ACTIVE
     * @tc.level     Level 0
     */
    it('NavDestinationStateActive_0100', Level.LEVEL0, async (done: Function) => {
      console.info('NavDestinationStateActive_0100 START');
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('pushId'));
      await btn1.click();
      console.info('[NavDestinationStateActive_0100] btn1 click');
      await driver.delayMs(3000);

      let testVal: number = AppStorage.get('ON_ACTIVE') as number
      console.log('NavDestinationStateActive_0100 on ON_ACTIVE value : ' + testVal)
      expect(testVal).assertEqual(8);
      AppStorage.delete('ON_ACTIVE');
      console.log('[NavDestinationStateActive_0100] END');
      done();
    });

    /**
     * @tc.number    NavDestinationStateActive_0200
     * @tc.name      NavDestinationStateActive_0200
     * @tc.desc      Test NavDestinationState ON_INACTIVE
     * @tc.level     Level 0
     */
    it('NavDestinationStateActive_0200', Level.LEVEL0, async (done: Function) => {
      console.info('NavDestinationStateActive_0200 START');
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('backId'));
      await btn1.click();
      console.info('[NavDestinationStateActive_0200] btn1 click');
      await driver.delayMs(3000);  

      let testVal2: number = AppStorage.get('ON_INACTIVE') as number
      console.log('NavDestinationStateActive_0200 on ON_INACTIVE value : ' + testVal2)
      expect(testVal2).assertEqual(9);
      AppStorage.delete('ON_INACTIVE');
      console.log('[NavDestinationStateActive_0200] END');
      done();
    });

  })
}