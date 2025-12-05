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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function QueryRouterPageInfoUndefined() {
  describe('QueryRouterPageInfoUndefined', () => {
    beforeEach(async (done: Function) => {
      console.info("QueryRouterPageInfoUndefined beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/QueryRouterPageInfoUndefined',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get QueryRouterPageInfoUndefined state pages:" + JSON.stringify(pages));
        if (!("QueryRouterPageInfoUndefined" == pages.name)) {
          console.info("get QueryRouterPageInfoUndefined pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push QueryRouterPageInfoUndefined page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push QueryRouterPageInfoUndefined page error:" + err);
      }
      console.info("QueryRouterPageInfoUndefined beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("QueryRouterPageInfoUndefined after each called");
    });


    /**
     * @tc.number    QueryRouterPageInfoUndefined_0100
     * @tc.name      QueryRouterPageInfoUndefined_0100
     * @tc.desc      Test queryRouterPageInfo return undefined
     * @tc.level     Level 0
     */
    it('QueryRouterPageInfoUndefined_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('QueryRouterPageInfoUndefined_0100 START');
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.text('transitionModal'));
      await btn1.click()
      console.info('[QueryRouterPageInfoUndefined_0100] btn1 click');
      await CommonFunc.sleep(1000);

      let btn2 = await driver.findComponent(ON.text('queryRouterPageInfo'));
      await btn2.click()
      console.info('[QueryRouterPageInfoUndefined_0100] btn2 click');
      await CommonFunc.sleep(1000);

      expect(AppStorage.get('undefinedValue')).assertUndefined();
      AppStorage.delete('undefinedValue')

      console.info('QueryRouterPageInfoUndefined_0100 END');
      done();
    });
  })
}