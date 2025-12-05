/**
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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import {
  Component,
  Driver,
  UiWindow,
  ON,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';
import events_emitter from '@ohos.events.emitter';
import CommonFunc from '../../MainAbility/common/Common';

export default function ListScrollEdgeCallback() {
  describe('ListScrollEdgeCallback', () => {
    beforeEach(async (done: Function) => {
      console.info('text beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/ListScrollEdgeCallback',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get ListScrollEdgeCallback state pages:' + JSON.stringify(pages));
        if (!('ListScrollEdgeCallback' == pages.name)) {
          console.info('get ListScrollEdgeCallback state pages.name:' + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info('push ListScrollEdgeCallback page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push ListScrollEdgeCallback page error:' + JSON.stringify(err));
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('ListScrollEdgeCallback after each called');
    });

    /**
     * @tc.number ListScrollEdgeCallbackTest_0100
     * @tc.name Test the properties of components
     * @tc.desc Set attribute values
     * @tc.size MediumTest
     * @tc.type Function
     * @tc.level Level 1
     */
    it('ListScrollEdgeCallbackTest_0100', Level.LEVEL1, async (done: Function) => {
      console.info('ListScrollEdgeCallbackTest_0100 START');
      let driver1 = Driver.create();
      let button = await driver1.findComponent(ON.id('ListScrollEdgeCallback01'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('ListTextByScrollEdge01')
      console.info('[List_ReachStartAndEnd_0100] component obj1 is: ' + JSON.stringify(JSON.parse(strJson1)))
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('ScrollEdgeSuccess');
      await CommonFunc.sleep(1000);
      console.info('ListScrollEdgeCallbackTest_0100 END');
      done();
    });

  })
}