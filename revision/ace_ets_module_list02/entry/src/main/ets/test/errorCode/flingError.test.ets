/**
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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
import { Driver, Component, UiWindow, ON } from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';
import events_emitter from '@ohos.events.emitter';

export default function flingError() {
  describe('flingError', () => {
    beforeEach(async (done: Function) => {
      console.info("flingError beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/errorCode/flingError",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get flingError state pages:" + JSON.stringify(pages));
        if (!("flingError" == pages.name)) {
          console.info("get flingError pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push flingError page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push flingError page error:" + err);
      }
      console.info("flingError beforeEach end");
      done();
    });

    /**
     * @tc.number    flingError_001
     * @tc.name      flingError_0100
     * @tc.desc      401 error code for the fling attribute.
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('flingError_0100', Level.LEVEL0, async (done: Function) => {
      console.info('[flingError_0100] START');
      let driver = await Driver.create()
      let btn1: Component = await driver.findComponent(ON.id('fling_401'));
      await btn1.click();
      await CommonFunc.sleep(1500);
      let strJson = getInspectorByKey('fling_err_text');
      console.info("[flingError_0100] component obj1 is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[flingError_0100] content: " + JSON.stringify(JSON.parse(strJson).$attrs.content));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('fling: 401');
      console.info('[flingError_0100] END');
      done();
    });

    /**
     * @tc.number    flingError_002
     * @tc.name      flingError_0200
     * @tc.desc      100004 error code for the fling attribute.
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('flingError_0200', Level.LEVEL0, async (done: Function) => {
      console.info('[flingError_0200] START');
      let driver = await Driver.create()
      let btn1: Component = await driver.findComponent(ON.id('fling_100004'));
      await btn1.click();
      await CommonFunc.sleep(1500);
      let strJson = getInspectorByKey('fling_err_text');
      console.info("[flingError_0200] component obj1 is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[flingError_0200] content: " + JSON.stringify(JSON.parse(strJson).$attrs.content));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('fling: "100004"');
      console.info('[flingError_0200] END');
      done();
    });
  })
}
