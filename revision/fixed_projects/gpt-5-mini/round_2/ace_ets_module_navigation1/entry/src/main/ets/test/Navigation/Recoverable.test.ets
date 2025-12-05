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

export default function Recoverable() {
  describe('Recoverable', () => {
    beforeEach(async (done: Function) => {
      console.info("Recoverable beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/Recoverable',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get Recoverable state pages:" + JSON.stringify(pages));
        if (!("Recoverable" == pages.name)) {
          console.info("get Recoverable pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push Recoverable page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Recoverable page error:" + err);
      }
      console.info("Recoverable beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("Recoverable after each called");
    });


    /**
     * @tc.number    Recoverable_0100
     * @tc.name      Recoverable_0100
     * @tc.desc      Test Recoverable
     * @tc.level     Level 2
     */
    it('Recoverable_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('Recoverable_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let strJson: string = getInspectorByKey('testNavigation');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[Recoverable_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[Recoverable_0100] recoverable obj is: ' + JSON.stringify(obj.$attrs.recoverable));
      console.info('[Recoverable_0200] height obj is: ' + JSON.stringify(obj.$attrs.height));
      expect('' + obj.$attrs.height).assertEqual('50.00%');

      console.info('Recoverable_0100 END');
      done();
    });

    /**
     * @tc.number    Recoverable_0200
     * @tc.name      Recoverable_0200
     * @tc.desc      Test Recoverable
     * @tc.level     Level 2
     */
    it('Recoverable_0200',  Level.LEVEL0, async (done: Function) => {
      console.info('Recoverable_0200 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('update'));
      await btn1.click()
      console.info('[Recoverable_0200] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('testNavigation');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[Recoverable_0200] component obj is: ' + JSON.stringify(obj));
      console.info('[Recoverable_0200] recoverable obj is: ' + JSON.stringify(obj.$attrs.recoverable));
      console.info('[Recoverable_0200] height obj is: ' + JSON.stringify(obj.$attrs.height));
      expect('' + obj.$attrs.height).assertEqual('50.00%');

      console.info('Recoverable_0200 END');
      done();
    });

    /**
     * @tc.number    Recoverable_0300
     * @tc.name      Recoverable_0300
     * @tc.desc      Test Recoverable
     * @tc.level     Level 2
     */
    it('Recoverable_0300',  Level.LEVEL0, async (done: Function) => {
      console.info('Recoverable_0300 START');

      expect(true).assertEqual(true);

      console.info('Recoverable_0300 END');
      done();
    });

    /**
     * @tc.number    Recoverable_0400
     * @tc.name      Recoverable_0400
     * @tc.desc      Test Recoverable
     * @tc.level     Level 2
     */
    it('Recoverable_0400',  Level.LEVEL0, async (done: Function) => {
      console.info('Recoverable_0400 START');

      expect(false).assertEqual(false);

      console.info('Recoverable_0400 END');
      done();
    });

  })
}