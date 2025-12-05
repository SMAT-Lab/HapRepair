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

export default function StepperModifierPage() {
  describe('StepperModifierPage', () => {
    beforeEach(async (done: Function) => {
      console.info('StepperModifierPage beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Stepper/StepperModifierPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get StepperModifierPage state pages:' + JSON.stringify(pages));
        if (!('StepperModifierPage' == pages.name)) {
          console.info('get StepperModifierPage pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push StepperModifierPage page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push StepperModifierPage page error:' + err);
      }
      console.info('StepperModifierPage beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('StepperModifierPage after each called');
    });


    /**
     * @tc.number    StepperModifierPage_0100
     * @tc.name      StepperModifierPage_0100
     * @tc.desc      Test onSkip
     * @tc.level     Level 0
     */
    it('StepperModifierPage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('StepperModifierPage_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let changeId1: Component = await driver.findComponent(ON.id('changeId1'));
      await changeId1.click()
      console.info('[StepperModifierPage_0100] changeId1 click');
      await driver.delayMs(1000);

      let StepperNext1: Component = await driver.findComponent(ON.text('StepperNext1'));
      await StepperNext1.click()
      console.info('[StepperModifierPage_0100] StepperNext1 click');
      await driver.delayMs(2000);

      let currentIndex: number = AppStorage.get('currentIndex') as number
      console.log('[StepperModifierPage_0100] currentIndex value : ' + currentIndex)
      expect(currentIndex).assertEqual(3);
      AppStorage.delete('currentIndex');

      console.info('StepperModifierPage_0100 END');
      done();
    });

    /**
     * @tc.number    StepperModifierPage_0200
     * @tc.name      StepperModifierPage_0200
     * @tc.desc      Test onFinish
     * @tc.level     Level 0
     */
    it('StepperModifierPage_0200', Level.LEVEL0, async (done: Function) => {
      console.info('StepperModifierPage_0200 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let fourFinish: Component = await driver.findComponent(ON.text('fourFinish'));
      await fourFinish.click()
      console.info('[StepperModifierPage_0200] fourFinish click');
      await driver.delayMs(2000);

      let currentIndex: number = AppStorage.get('currentIndex') as number
      console.log('[StepperModifierPage_0200] currentIndex value : ' + currentIndex)
      expect(currentIndex).assertEqual(0);
      AppStorage.delete('currentIndex');

      console.info('StepperModifierPage_0200 END');
      done();
    });

    /**
     * @tc.number    StepperModifierPage_0300
     * @tc.name      StepperModifierPage_0300
     * @tc.desc      Test next onchange
     * @tc.level     Level 0
     */
    it('StepperModifierPage_0300', Level.LEVEL0, async (done: Function) => {
      console.info('StepperModifierPage_0300 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let changeId1: Component = await driver.findComponent(ON.id('changeId1'));
      await changeId1.click()
      console.info('[StepperModifierPage_0300] changeId1 click');
      await driver.delayMs(1000);

      let StepperNext1: Component = await driver.findComponent(ON.text('StepperNext1'));
      await StepperNext1.click()
      console.info('[StepperModifierPage_0300] StepperNext1 click');
      await driver.delayMs(1000);

      let currentIndex: number = AppStorage.get('currentIndex') as number
      console.log('[StepperModifierPage_0200] currentIndex value : ' + currentIndex)
      expect(currentIndex).assertEqual(1);
      AppStorage.delete('currentIndex');

      let pendingIndex: number = AppStorage.get('pendingIndex') as number
      console.log('[StepperModifierPage_0200] pendingIndex value : ' + pendingIndex)
      expect(pendingIndex).assertEqual(1);
      AppStorage.delete('pendingIndex');

      console.info('StepperModifierPage_0300 END');
      done();
    });

    /**
     * @tc.number    StepperModifierPage_0400
     * @tc.name      StepperModifierPage_0400
     * @tc.desc      Test Previous
     * @tc.level     Level 0
     */
    it('StepperModifierPage_0400', Level.LEVEL0, async (done: Function) => {
      console.info('StepperModifierPage_0400 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let previousBtn: Component = await driver.findComponent(ON.text('StepperPrevious'));
      await previousBtn.click()
      console.info('[StepperModifierPage_0400] previousBtn click');
      await driver.delayMs(1000);

      let currentIndex: number = AppStorage.get('currentIndex') as number
      console.log('[StepperModifierPage_0400] currentIndex value : ' + currentIndex)
      expect(currentIndex).assertEqual(0);
      AppStorage.delete('currentIndex');

      let pendingIndex: number = AppStorage.get('pendingIndex') as number
      console.log('[StepperModifierPage_0400] pendingIndex value : ' + pendingIndex)
      expect(pendingIndex).assertEqual(0);
      AppStorage.delete('pendingIndex');

      console.info('StepperModifierPage_0400 END');
      done();
    });

  })
}