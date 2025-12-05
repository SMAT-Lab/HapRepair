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

export default function UIContextNavDestinationUpdate() {
  describe('UIContextNavDestinationUpdate', () => {
    beforeEach(async (done: Function) => {
      console.info('UIContextNavDestinationUpdate beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/UIContext/UIContextNavDestinationUpdate',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get UIContextNavDestinationUpdate state pages:' + JSON.stringify(pages));
        if (!('UIContextNavDestinationUpdate' == pages.name)) {
          console.info('get UIContextNavDestinationUpdate pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push UIContextNavDestinationUpdate page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push UIContextNavDestinationUpdate page error:' + err);
      }
      console.info('UIContextNavDestinationUpdate beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('UIContextNavDestinationUpdate after each called');
    });


    /**
     * @tc.number    UIContextNavDestinationUpdate_0100
     * @tc.name      UIContextNavDestinationUpdate_0100
     * @tc.desc      Test uniqueId
     * @tc.level     Level 0
     */
    it('UIContextNavDestinationUpdate_0100', Level.LEVEL0, async (done: Function) => {
      console.info('UIContextNavDestinationUpdate_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let pushOneBtn: Component = await driver.findComponent(ON.id('pushOneId'));
      await pushOneBtn.click()
      console.info('[UIContextNavDestinationUpdate_0100] pushOneBtn click');
      await driver.delayMs(1000);

      let uniqueId: string = AppStorage.get('uniqueId') as string
      console.log('[UIContextNavDestinationUpdate_0100] uniqueId value : ' + uniqueId)
      expect(uniqueId !== '').assertTrue();
      AppStorage.delete('uniqueId');

      console.info('UIContextNavDestinationUpdate_0100 END');
      done();
    });

    /**
     * @tc.number    UIContextNavDestinationUpdate_0200
     * @tc.name      UIContextNavDestinationUpdate_0200
     * @tc.desc      Test uniqueId, uiObserver.on
     * @tc.level     Level 0
     */
    it('UIContextNavDestinationUpdate_0200', Level.LEVEL0, async (done: Function) => {
      console.info('UIContextNavDestinationUpdate_0200 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let onBtn: Component = await driver.findComponent(ON.id('onId'));
      await onBtn.click()
      console.info('[UIContextNavDestinationUpdate_0200] onBtn click');
      await driver.delayMs(1000);

      let pushTwoBtn: Component = await driver.findComponent(ON.id('pushTwoId'));
      await pushTwoBtn.click()
      console.info('[UIContextNavDestinationUpdate_0200] pushTwoBtn click');
      await driver.delayMs(3000);

      let navDestinationName: string = AppStorage.get('navDestinationName') as string
      console.log('[UIContextNavDestinationUpdate_0200] navDestinationName value : ' + navDestinationName)
      expect(navDestinationName).assertEqual('pageTwo')
      AppStorage.delete('navDestinationName');

      let backBtn: Component = await driver.findComponent(ON.id('backId'));
      await backBtn.click()
      console.info('[UIContextNavDestinationUpdate_0200] backBtn click');
      await driver.delayMs(1000);

      console.info('UIContextNavDestinationUpdate_0200 END');
      done();
    });

    /**
     * @tc.number    UIContextNavDestinationUpdate_0300
     * @tc.name      UIContextNavDestinationUpdate_0300
     * @tc.desc      Test uniqueId, uiObserver.off
     * @tc.level     Level 0
     */
    it('UIContextNavDestinationUpdate_0300', Level.LEVEL0, async (done: Function) => {
      console.info('UIContextNavDestinationUpdate_0300 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let offBtn: Component = await driver.findComponent(ON.id('offId'));
      await offBtn.click()
      console.info('[UIContextNavDestinationUpdate_0300] offBtn click');
      await driver.delayMs(1000);

      AppStorage.delete('navDestinationName');
      await driver.delayMs(1000);

      let pushTwoBtn: Component = await driver.findComponent(ON.id('pushTwoId'));
      await pushTwoBtn.click()
      console.info('[UIContextNavDestinationUpdate_0300] pushTwoBtn click');
      await driver.delayMs(3000);

      let navDestinationName: Object | undefined = AppStorage.get('navDestinationName')
      console.log('[UIContextNavDestinationUpdate_0300] navDestinationName value : ' + navDestinationName)
      expect(navDestinationName).assertEqual(undefined)

      console.info('UIContextNavDestinationUpdate_0300 END');
      done();
    });

  })
}