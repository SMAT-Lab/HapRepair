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

export default function SystemTransition() {
  describe('SystemTransition', () => {
    beforeEach(async (done: Function) => {
      console.info('SystemTransition beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/SystemTransition',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get SystemTransition state pages:' + JSON.stringify(pages));
        if (!('SystemTransition' == pages.name)) {
          console.info('get SystemTransition pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(1000);
          console.info('push SystemTransition page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push SystemTransition page error:' + err);
      }
      console.info('SystemTransition beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('SystemTransition after each called');
    });


    /**
     * @tc.number    SystemTransition_0100
     * @tc.name      SystemTransition_0100
     * @tc.desc      Test systemTransition and NavigationSystemTransitionType.TITLE and NavigationSystemTransitionType.CONTENT
     * @tc.level     Level 2
     */
    it('SystemTransition_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('SystemTransition_0100 START');
      await CommonFunc.sleep(200);

      let driver = Driver.create();
      await CommonFunc.sleep(500);

      let btn1 = await driver.findComponent(ON.id('pageOneId'));
      await btn1.click()
      console.info('[SystemTransition_0100] btn1 click');
      await CommonFunc.sleep(200);

      let strJson1: string = getInspectorByKey('pageOne');
      let obj1: Record<string, ESObject> = JSON.parse(strJson1);
      console.info('[SystemTransition_0100] component page one obj is: ' + JSON.stringify(obj1));
      console.info('[SystemTransition_0100] content page one obj is: ' + JSON.stringify(obj1.$attrs.content));
      expect(obj1.$attrs.content).assertEqual('This is page one');

      let btn2 = await driver.findComponent(ON.id('popId'));
      await btn2.click()
      console.info('[SystemTransition_0100] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('pageTwoId'));
      await btn3.click()
      console.info('[SystemTransition_0100] btn3 click');
      await CommonFunc.sleep(200);

      let strJson2: string = getInspectorByKey('pageTwo');
      let obj2: Record<string, ESObject> = JSON.parse(strJson2);
      console.info('[SystemTransition_0100] component page two obj is: ' + JSON.stringify(obj2));
      console.info('[SystemTransition_0100] content page two obj is: ' + JSON.stringify(obj2.$attrs.content));
      expect(obj2.$attrs.content).assertEqual('This is page two');

      console.info('SystemTransition_0100 END');
      done();
    });

  })
}