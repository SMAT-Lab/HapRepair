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
import { MessageManager, Callback } from '../../MainAbility/utils/MessageManager';
import { Driver, ON } from '@ohos.UiTest';

export default function Router4() {
  describe('Router4', () => {
    beforeEach(async (done: Function) => {
      console.info("GridFrictionPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/pageA',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get " +
          " state pages:" + JSON.stringify(pages));
        if (!("pageA" == pages.name)) {
          console.info("get Router4 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push Router4 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Router4 page error:" + err);
      }
      console.info("Router4 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back()
      console.info("Router4 after each called");
    });
    /**
     * @tc.number Router4_0100
     * @tc.name Router4_0100
     * @tc.desc Repeatedly jump to the page
     * @tc.level: Level 2
     * @tc.type :
     * @tc.size :
     */

    it('Router4_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[Router4_0100] START');


      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(500);

      let strJson1 = getInspectorByKey('pageA_name');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('pageA');
      await CommonFunc.sleep(500);

      console.info('obj111111' + JSON.parse(strJson1).$attrs.content);

      let strJson2 = getInspectorByKey('pageA_length');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('2');
      await CommonFunc.sleep(500);

      console.info('obj22222' + JSON.parse(strJson2).$attrs.content);

      await CommonFunc.sleep(500);
      let btn = await driver.findComponent(ON.id('skip_B'));
      await btn.click();
      await CommonFunc.sleep(1000);


      let strJson3 = getInspectorByKey('name_B');
      expect(JSON.parse(strJson3).$attrs.content).assertEqual('pageB');

      console.info('obj3333333' + JSON.parse(strJson3).$attrs.content);
      await CommonFunc.sleep(500);

      let strJson4 = getInspectorByKey('name_length');
      expect(JSON.parse(strJson4).$attrs.content).assertEqual('3');

      console.info('obj444444' + JSON.parse(strJson4).$attrs.content);
      await CommonFunc.sleep(500);


      await CommonFunc.sleep(500);
      let btn2 = await driver.findComponent(ON.id('clear_B'));
      await btn2.click();
      await CommonFunc.sleep(1000);

      let strJson5 = getInspectorByKey('name_B');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('pageA');
      console.info('obj555555' + JSON.parse(strJson1).$attrs.content)
      await CommonFunc.sleep(500);

      let strJson6 = getInspectorByKey('name_length');
      expect(JSON.parse(strJson6).$attrs.content).assertEqual('3');
      console.info('obj66666' + JSON.parse(strJson6).$attrs.content)
      await CommonFunc.sleep(500);


      await CommonFunc.sleep(500);
      let btn3 = await driver.findComponent(ON.id('page_c'));
      await btn3.click();
      await CommonFunc.sleep(1000);


      let strJson7 = getInspectorByKey('name_C');
      expect(JSON.parse(strJson7).$attrs.content).assertEqual('pageC');

      console.info('obj7777777' + JSON.parse(strJson7).$attrs.content)
      await CommonFunc.sleep(500);

      let strJson8 = getInspectorByKey('length_C');
      expect(JSON.parse(strJson8).$attrs.content).assertEqual('2');

      console.info('obj88888' + JSON.parse(strJson8).$attrs.content)
      await CommonFunc.sleep(100);
      done();
    });
  })
}
