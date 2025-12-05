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

export default function childrenSizeError() {
  describe('childrenSizeError', () => {
    beforeEach(async (done: Function) => {
      console.info("childrenSizeError beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/errorCode/childrenSizeError",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get childrenSizeError state pages:" + JSON.stringify(pages));
        if (!("childrenSizeError" == pages.name)) {
          console.info("get childrenSizeError pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push childrenSizeError page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push childrenSizeError page error:" + err);
      }
      console.info("childrenSizeError beforeEach end");
      done();
    });

    /**
     * @tc.number    childrenSizeError_001
     * @tc.name      childrenSizeError_0100
     * @tc.desc      Error code for splice property of childrenMainSize object.
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('childrenSizeError_0100', Level.LEVEL0, async (done: Function) => {
      console.info('[childrenSizeError_0100] START');
      let driver = await Driver.create()
      let btn1: Component = await driver.findComponent(ON.id('splice_btn_401'));
      await btn1.click();
      await CommonFunc.sleep(1500);
      let strJson = getInspectorByKey('splice_err_401');
      console.info("[childrenSizeError_0100] component obj1 is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[childrenSizeError_0100] content: " + JSON.stringify(JSON.parse(strJson).$attrs.content));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('spliceError: 401');
      console.info('[childrenSizeError_0100] END');
      done();
    });

    /**
     * @tc.number    childrenSizeError_002
     * @tc.name      childrenSizeError_0200
     * @tc.desc      Error code for update property of childrenMainSize object.
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('childrenSizeError_0200', Level.LEVEL0, async (done: Function) => {
      console.info('[childrenSizeError_0200] START');
      let driver = await Driver.create()
      let btn1: Component = await driver.findComponent(ON.id('update_btn_401'));
      await btn1.click();
      await CommonFunc.sleep(1500);
      let strJson = getInspectorByKey('update_err_401');
      console.info("[childrenSizeError_0200] component obj1 is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[childrenSizeError_0200] content: " + JSON.stringify(JSON.parse(strJson).$attrs.content));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('updateError: 401');
      console.info('[childrenSizeError_0200] END');
      done();
    });

    /**
     * @tc.number    childrenSizeError_003
     * @tc.name      childrenSizeError_0300
     * @tc.desc      Error code for childDefaultSize property of childrenMainSize object.
     * @tc.level     Level 0
     * @tc.type
     * @tc.size
     */
    it('childrenSizeError_0300', Level.LEVEL0, async (done: Function) => {
      console.info('[childrenSizeError_0300] START');
      let driver = await Driver.create()
      let btn1: Component = await driver.findComponent(ON.id('childDefaultSize_btn_401'));
      await btn1.click();
      await CommonFunc.sleep(1500);
      let strJson = getInspectorByKey('childDefaultSize_err_401');
      console.info("[childrenSizeError_0300] component obj1 is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[childrenSizeError_0300] content: " + JSON.stringify(JSON.parse(strJson).$attrs.content));
      expect(JSON.parse(strJson).$attrs.content).assertEqual('childDefaultSize: 401');
      console.info('[childrenSizeError_0300] END');
      done();
    });
  })
}
