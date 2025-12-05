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
import CommonFunc from '../../MainAbility/utils/Common';
import {
  UiComponent,
  UiDriver,
  Component,
  Driver,
  UiWindow,
  ON,
  BY,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';

export default function router_back10() {
  describe('router_back10', () => {
    beforeEach(async (done: Function) => {
      console.info("router_back10 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/router_back10',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get router_back10 state pages:" + JSON.stringify(pages));
        if (!("router_back10" == pages.name)) {
          console.info("get router_back10 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push router_back10 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push router_back10 page error:" + err);
      }
      console.info("router_back10 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("router_back10 after each called");
    });

    /**
     * @tc.number SUB_ACE_UI_ABILITY_ROUTER_0310
     * @tc.name testBack
     * @tc.desc Return specified page by index value
     * @tc.level Level 3
     * @tc.type
     * @tc.size
     */
    it('router_back10_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[router_back10_0100] START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.id('router_back10_btn'));
      await btn1.click();
      await CommonFunc.sleep(2000);

      let text1 = getInspectorByKey('router_back10_text');
      expect(JSON.parse(text1).$attrs.content).assertEqual('back10');

      console.info('[router_back10_0100] END');
      done();
    });

  })
}
