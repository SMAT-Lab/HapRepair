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

export default function StepperModifierTest() {
  describe('StepperModifierTest', () => {
    beforeEach(async (done: Function) => {
      console.info('StepperModifierTest beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Stepper/StepperModifierTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get StepperModifierTest state pages:' + JSON.stringify(pages));
        if (!('StepperModifierTest' == pages.name)) {
          console.info('get StepperModifierTest pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push StepperModifierTest page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push StepperModifierTest page error:' + err);
      }
      console.info('StepperModifierTest beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('StepperModifierTest after each called');
    });


    /**
     * @tc.number    StepperModifierTest_0100
     * @tc.name      StepperModifierTest_0100
     * @tc.desc      Test onSkip
     * @tc.level     Level 0
     */
    it('StepperModifierTest_0100', Level.LEVEL0, async (done: Function) => {
      console.info('StepperModifierTest_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(1000);

      let strJson: string = getInspectorByKey('pageThree');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[NavigationJsunit_0400] content is: ' + obj.$attrs.content);
      expect(obj.$attrs.content).assertEqual('Page Three');
      console.info('StepperModifierTest_0100 END');
      done();
    });

  })
}