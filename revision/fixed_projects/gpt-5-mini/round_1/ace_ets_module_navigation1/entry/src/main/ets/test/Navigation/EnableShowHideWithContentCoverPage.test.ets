/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

export default function EnableShowHideWithContentCoverPage() {
  describe('EnableShowHideWithContentCoverPage', () => {
    beforeEach(async (done: Function) => {
      console.info('EnableShowHideWithContentCoverPage beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/EnableShowHideWithContentCoverPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get EnableShowHideWithContentCoverPage state pages:' + JSON.stringify(pages));
        if (!('EnableShowHideWithContentCoverPage' == pages.name)) {
          console.info('get EnableShowHideWithContentCoverPage pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push EnableShowHideWithContentCoverPage page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push EnableShowHideWithContentCoverPage page error:' + err);
      }
      console.info('EnableShowHideWithContentCoverPage beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('EnableShowHideWithContentCoverPage after each called');
    });


    /**
     * @tc.number    EnableShowHideWithContentCoverPage_0100
     * @tc.name      EnableShowHideWithContentCoverPage_0100
     * @tc.desc      Test EnableShowHideWithContentCover and VisibilityChangeReason.TRANSITION
     * @tc.level     Level 0
     */
    it('EnableShowHideWithContentCoverPage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('EnableShowHideWithContentCoverPage_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.text('pushOne'));
      await btn1.click();
      console.log('[EnableShowHideWithContentCoverPage_0100] btn1 click');
      await driver.delayMs(3000);
      let testValue1: string = AppStorage.get('testValue') as string;
      expect(testValue1).assertEqual('0');
      AppStorage.delete('testValue');

      let btn2: Component = await driver.findComponent(ON.text('back'));
      await btn2.click();
      console.log('[EnableShowHideWithContentCoverPage_0100] btn2 click');
      await driver.delayMs(3000);
      let testValue2: string = AppStorage.get('testValue') as string;
      expect(testValue2).assertEqual('0');
      AppStorage.delete('testValue');
      console.info('EnableShowHideWithContentCoverPage_0100 END');
      done();
    });

    /**
     * @tc.number    EnableShowHideWithContentCoverPage_0200
     * @tc.name      EnableShowHideWithContentCoverPage_0200
     * @tc.desc      Test EnableShowHideWithContentCover and VisibilityChangeReason.CONTENT_COVER
     * @tc.level     Level 0
     */
    it('EnableShowHideWithContentCoverPage_0200', Level.LEVEL0, async (done: Function) => {
      console.info('EnableShowHideWithContentCoverPage_0200 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.text('pushOne'));
      await btn1.click();
      console.log('[EnableShowHideWithContentCoverPage_0200] btn1 click');
      await driver.delayMs(1000);

      let btn2: Component = await driver.findComponent(ON.text('openModal'));
      await btn2.click();
      console.log('[EnableShowHideWithContentCoverPage_0200] btn2 click');
      await driver.delayMs(3000);
      let testValue1: string = AppStorage.get('testValue') as string;
      expect(testValue1).assertEqual('1');
      AppStorage.delete('testValue');

      let btn3: Component = await driver.findComponent(ON.text('closeModal'));
      await btn3.click();
      console.log('[EnableShowHideWithContentCoverPage_0200] btn3 click');
      await driver.delayMs(3000);
      let testValue2: string = AppStorage.get('testValue') as string;
      expect(testValue2).assertEqual('1');
      AppStorage.delete('testValue');

      let backBtn: Component = await driver.findComponent(ON.text('back'));
      await backBtn.click();
      console.log('[EnableShowHideWithContentCoverPage_0200] backBtn click');
      await driver.delayMs(1000);
      console.info('EnableShowHideWithContentCoverPage_0200 END');
      done();
    });

    /**
     * @tc.number    EnableShowHideWithContentCoverPage_0300
     * @tc.name      EnableShowHideWithContentCoverPage_0300
     * @tc.desc      Test EnableShowHideWithContentCover false and VisibilityChangeReason.CONTENT_COVER
     * @tc.level     Level 0
     */
    it('EnableShowHideWithContentCoverPage_0300', Level.LEVEL0, async (done: Function) => {
      console.info('EnableShowHideWithContentCoverPage_0300 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let enbaleBtn: Component = await driver.findComponent(ON.text('changeEnableValue'));
      await enbaleBtn.click();
      console.log('[EnableShowHideWithContentCoverPage_0300] enbaleBtn click');
      await driver.delayMs(1000);

      let btn1: Component = await driver.findComponent(ON.text('pushOne'));
      await btn1.click();
      console.log('[EnableShowHideWithContentCoverPage_0300] btn1 click');
      await driver.delayMs(1000);

      let btn2: Component = await driver.findComponent(ON.text('openModal'));
      await btn2.click();
      console.log('[EnableShowHideWithContentCoverPage_0300] btn2 click');
      await driver.delayMs(3000);
      let testValue1: string | undefined = AppStorage.get('testValue')
      expect(testValue1).assertEqual(undefined);

      let btn3: Component = await driver.findComponent(ON.text('closeModal'));
      await btn3.click();
      console.log('[EnableShowHideWithContentCoverPage_0300] btn3 click');
      await driver.delayMs(3000);
      let testValue2: string | undefined = AppStorage.get('testValue');
      expect(testValue2).assertEqual(undefined);
      console.info('EnableShowHideWithContentCoverPage_0300 END');
      done();
    });

  })
}