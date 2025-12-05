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

export default function RouterCall() {
  describe('RouterCall', () => {
    beforeEach(async (done: Function) => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ohosRouter/RouterCall',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get router state success ' + JSON.stringify(pages));
        if (!('RouterCall' == pages.name)) {
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
      console.info('RouterCall after each called');
    });

    /**
     * @tc.number    RouterCall_0400
     * @tc.name      RouterCall_0400
     * @tc.desc      Test Router pushUrl callback 401
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterCall_0400', Level.LEVEL2, async (done: Function) => {
      console.info('RouterCall_0400 START');
      await Utils.sleep(1000);

      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('401Btn2'));
      await btn1.click()
      console.info('[RouterCall_0400] btn1 click');
      await driver.delayMs(2000);

      let code: string | undefined = AppStorage.get('401Code')
      expect(code).assertEqual('401');
      AppStorage.delete('401Code'); 
      console.info('[RouterCall_0400] END');
      done();
    });

    /**
     * @tc.number    RouterCall_0500
     * @tc.name      RouterCall_0500
     * @tc.desc      Test Router pushUrl callback 100002
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterCall_0500', Level.LEVEL2, async (done: Function) => {
      console.info('RouterCall_0500 START');
      await Utils.sleep(1000);

      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.id('100002Btn2'));
      await btn1.click()
      console.info('[RouterCall_0500] btn1 click');
      await driver.delayMs(2000);

      let code: string | undefined = AppStorage.get('100002Code')
      expect(code).assertEqual('100002');
      AppStorage.delete('100002Code'); 
      console.info('[RouterCall_0500] END');
      done();
    });

    /**
     * @tc.number    RouterCall_0600
     * @tc.name      RouterCall_0600
     * @tc.desc      Test Router pushUrl 100003
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterCall_0600', Level.LEVEL2, async (done: Function) => {
      console.info('RouterCall_0600 START');
      await Utils.sleep(1000);

      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.text('routerCallbackPush100003'));
      await btn1.click()
      console.info('[RouterCall_0600] btn1 click');
      await driver.delayMs(3000);

      let code: string | undefined = AppStorage.get('100003Code')
      expect(code).assertEqual('100003');
      AppStorage.delete('100003Code'); 
      console.info('[RouterCall_0600] END');
      done();
    });


  })
}
