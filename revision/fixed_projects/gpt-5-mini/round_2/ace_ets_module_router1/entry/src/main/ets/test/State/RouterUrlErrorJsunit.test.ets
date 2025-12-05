/**
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

import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/utils/Common';

export default function RouterUrlErrorJsunit() {
  describe('RouterUrlErrorJsunit', () =>{
    beforeEach(async  (done:Function) =>{
      console.info("RouterUrlErrorJsunit beforeEach start");
      let options:router.RouterOptions = {
        url: 'MainAbility/pages/State/RouterUrlError',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get RouterUrlErrorJsunit state pages:" + JSON.stringify(pages));
        if (!("RouterUrlError" == pages.name)) {
          console.info("get RouterUrlErrorJsunit state pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          console.info("push RouterUrlErrorJsunit success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push RouterUrlErrorJsunit page error:" + err);
        expect().assertFail();
      }
      done();
    });

    /**
     * @tc.number    SUB_ACE_BASICABILITY_ROUTERPAGE_009
     * @tc.name      RouterUrlErrorJsunit_0100
     * @tc.desc      The route input parameter is abnormally redirected
     * @tc.level:    Level 2
     * @tc.type :
     * @tc.size :
     */
    it('RouterUrlErrorJsunit_0100', Level.LEVEL2, async  (done:Function) =>{
      console.info('RouterUrlErrorJsunit_0100 START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('RouterUrlErrorButton'));
      await button.click();
      await CommonFunc.sleep(2000);
      let strJson1 = getInspectorByKey('RouterUrlErrorText');
      console.info('RouterUrlErrorJsunit_0100 component obj is: ' + JSON.parse(strJson1).$attrs.content);
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('start');
      await CommonFunc.sleep(1000);
      done();
    });

  })
}