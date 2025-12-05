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

export default function SystemTransition2() {
  describe('SystemTransition2', () => {
    beforeEach(async (done: Function) => {
      console.info('SystemTransition2 beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/SystemTransition2',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get SystemTransition2 state pages:' + JSON.stringify(pages));
        if (!('SystemTransition2' == pages.name)) {
          console.info('get SystemTransition2 pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(1000);
          console.info('push SystemTransition2 page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push SystemTransition2 page error:' + err);
      }
      console.info('SystemTransition2 beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('SystemTransition2 after each called');
    });


    /**
     * @tc.number    SystemTransition2_0100
     * @tc.name      SystemTransition2_0100
     * @tc.desc      Test SystemTransition2 and NavigationSystemTransitionType.DEFAULT
     * @tc.level     Level 2
     */
    it('SystemTransition2_0100', Level.LEVEL0, async (done: Function) => {
      console.info('SystemTransition2_0100 START');
      await CommonFunc.sleep(200);

      let driver = Driver.create();
      await CommonFunc.sleep(500);

      let btn1 = await driver.findComponent(ON.id('pushOneId'));
      await btn1.click()
      console.info('[SystemTransition2_0100] btn1 click');
      await CommonFunc.sleep(1000);

      let strJson = getInspectorByKey('pageOneId');
      let obj: ESObject = JSON.parse(strJson);
      console.info('[SystemTransition2_0100] component obj is: ' + JSON.stringify(obj));
      console.log('[SystemTransition2_0100] systemTransition is: ' + JSON.stringify(obj.$attrs.systemTransition));
      expect(obj.$attrs.systemTransition).assertEqual('NavigationSystemTransitionType.DEFAULT');


      let btn2 = await driver.findComponent(ON.id('popId'));
      await btn2.click()
      console.info('[SystemTransition2_0100] btn2 click');
      await CommonFunc.sleep(1000);
      console.info('SystemTransition2_0100 END');
      done();
    });

    /**
     * @tc.number    SystemTransition2_0200
     * @tc.name      SystemTransition2_0200
     * @tc.desc      Test SystemTransition2 and NavigationSystemTransitionType.NONE
     * @tc.level     Level 2
     */
    it('SystemTransition2_0200', Level.LEVEL0, async (done: Function) => {
      console.info('SystemTransition2_0200 START');
      await CommonFunc.sleep(200);

      let driver = Driver.create();
      await CommonFunc.sleep(500);

      let btn1 = await driver.findComponent(ON.id('pushTwoId'));
      await btn1.click()
      console.info('[SystemTransition2_0200] btn1 click');
      await CommonFunc.sleep(1000);

      let strJson = getInspectorByKey('pageTwoId');
      let obj: ESObject = JSON.parse(strJson);
      console.info('[SystemTransition2_0200] component obj is: ' + JSON.stringify(obj));
      console.log('[SystemTransition2_0200] systemTransition is: ' + JSON.stringify(obj.$attrs.systemTransition));
      expect(obj.$attrs.systemTransition).assertEqual('NavigationSystemTransitionType.NONE');

      console.info('SystemTransition2_0200 END');
      done();
    });

  })
}