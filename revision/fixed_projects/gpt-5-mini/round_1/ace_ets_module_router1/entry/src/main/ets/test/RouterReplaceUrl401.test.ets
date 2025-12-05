/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import CommonFunc from '../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import events_emitter from '@ohos.events.emitter'

export default function RouterReplaceUrl401() {
  describe('RouterReplaceUrl401', () => {
    beforeEach(async (done: Function) => {
      console.info('RouterReplaceUrl401 beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/RouterReplaceUrl401',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get RouterReplaceUrl401 state pages:' + JSON.stringify(pages));
        if (!('RouterReplaceUrl401'== pages.name)) {
          console.info('get RouterReplaceUrl401 pages.name:' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push RouterReplaceUrl401 page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push RouterReplaceUrl401 page error:' + err);
      }
      console.info('RouterReplaceUrl401 beforeEach end');
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('RouterReplaceUrl401 after each called');
    });


    /**
     * @tc.number    RouterReplaceUrl401_0100
     * @tc.name      RouterReplaceUrl401_0100
     * @tc.desc      Test replaceUrl error code 401
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('RouterReplaceUrl401_0100', Level.LEVEL2, async (done: Function) => {
      console.info('RouterReplaceUrl401_0100 START');
      await CommonFunc.sleep(500);
      let driver = Driver.create();
      await driver.delayMs(500);

      let btn1: Component = await driver.findComponent(ON.id('401Btn2'));
      await btn1.click()
      console.info('[RouterReplaceUrl401_0100] btn1 click');
      await driver.delayMs(1000);

      let strJson: string = getInspectorByKey('id06');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[PushDestination401_0200] component obj is: ' + JSON.stringify(obj));
      console.info('[PushDestination401_0200] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('401');

      console.info('[RouterReplaceUrl401_0100]  END');
      done();
    });

  })
}