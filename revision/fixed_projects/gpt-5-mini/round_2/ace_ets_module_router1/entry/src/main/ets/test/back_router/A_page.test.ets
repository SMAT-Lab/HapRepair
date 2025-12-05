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
  PointerMatrix,
  UiDirection,
  MouseButton,
  UIElementInfo,
  UIEventObserver
} from '@ohos.UiTest';

export default function Params() {
  describe('Params', () => {
    beforeEach(async (done: Function) => {
      console.info("Params beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/back_router/A_page',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get Params state pages:" + JSON.stringify(pages));
        if (!("Params" == pages.name)) {
          console.info("get WaterFlow_WaterFlow pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push Params page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Params page error:" + err);
      }
      console.info("Params beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("Params after each called");
    });
    /**
     * @tc.number SUB_ACE_RICHEDITOR_Params_0100
     * @tc.name Params
     * @tc.desc set api
     * @tc.level Level 2
     * @tc.type
     * @tc.size
     */
    it('Params_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[Params_0100] START');
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('R_params'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('text_Params1');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('{"data1":{"data2":["message","message1"]}}');
      await CommonFunc.sleep(1000);
      done();
    });
  })
}
