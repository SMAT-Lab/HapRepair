/**
 * Copyright (c) 2025 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import Utils from '../../MainAbility/common/Utils'
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function RouterMode() {
  describe('RouterMode', () => {
    beforeEach(async (done: Function) => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ohosRouter/RouterMode',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get router state success ' + JSON.stringify(pages));
        if (!('RouterMode' == pages.name)) {
          console.info('get router state success ' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await Utils.sleep(2000);
          console.info('push router page success ' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push router page error: ' + err);
      }
      done()
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info('RouterMode after each called');
    });

    /**
     * @tc.number    RouterMode_0100
     * @tc.name      RouterMode_0100
     * @tc.desc      Test Router pushUrl 401
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterMode_0100', Level.LEVEL2, async (done: Function) => {
      console.info('RouterMode_0100 START');
      await Utils.sleep(1000);

      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('401Btn1'));
      await btn1.click()
      console.info('[RouterMode_0100] btn1 click');
      await driver.delayMs(2000);

      let code: string | undefined = AppStorage.get('401Code')
      expect(code).assertEqual('401');
      AppStorage.delete('401Code'); 
      console.info('[RouterMode_0100] END');
      done();
    });

    /**
     * @tc.number    RouterMode_0200
     * @tc.name      RouterMode_0200
     * @tc.desc      Test Router pushUrl 100002
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterMode_0200', Level.LEVEL2, async (done: Function) => {
      console.info('RouterMode_0200 START');
      await Utils.sleep(1000);

      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('100002Btn1'));
      await btn1.click()
      console.info('[RouterMode_0200] btn1 click');
      await driver.delayMs(2000);

      let code: string | undefined = AppStorage.get('100002Code')
      expect(code).assertEqual('100002');
      AppStorage.delete('100002Code'); 
      console.info('[RouterMode_0200] END');
      done();
    });

    /**
     * @tc.number    RouterMode_0300
     * @tc.name      RouterMode_0300
     * @tc.desc      Test Router pushUrl 100003
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterMode_0300', Level.LEVEL2, async (done: Function) => {
      console.info('RouterMode_0300 START');
      await Utils.sleep(1000);

      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.text('routerPushUrl100003'));
      await btn1.click()
      console.info('[RouterMode_0300] btn1 click');
      await driver.delayMs(3000);

      let code: string | undefined = AppStorage.get('100003Code')
      expect(code).assertEqual('100003');
      AppStorage.delete('100003Code'); 
      console.info('[RouterMode_0300] END');
      done();
    });

  })
}
