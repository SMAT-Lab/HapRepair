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

export default function Router3() {
  describe('Router3', () => {
    beforeEach(async (done: Function) => {
      console.info("GridFrictionPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/pageOne',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get " +
          " state pages:" + JSON.stringify(pages));
        if (!("pageOne" == pages.name)) {
          console.info("get Router3 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push Router3 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Router3 page error:" + err);
      }
      console.info("Router3 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back()
      console.info("Router3 after each called");
    });
    /**
     * @tc.number Router3_0100
     * @tc.name Router3_0100
     * @tc.desc Repeatedly jump to the page
     * @tc.level: Level 2
     * @tc.type :
     * @tc.size :
     */

    it('Router3_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[Router3_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(500);


      let strJson1 = getInspectorByKey('nameState');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('pageOne');
      console.info('obj1_Router3' + JSON.parse(strJson1).$attrs.content)
      await CommonFunc.sleep(500);

      let strJson2 = getInspectorByKey('getNum');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('2');
      console.info('obj2_Router3' + JSON.parse(strJson2).$attrs.content)
      await CommonFunc.sleep(500);


      await CommonFunc.sleep(500);
      let btn = await driver.findComponent(ON.id('page_Button'));
      await btn.click();
      await CommonFunc.sleep(1000);

      let strJson3 = getInspectorByKey('pageTwo_name');
      expect(JSON.parse(strJson3).$attrs.content).assertEqual('pageTwo');
      console.info('obj3_Router3' + JSON.parse(strJson3).$attrs.content)
      await CommonFunc.sleep(500);
      let strJson4 = getInspectorByKey('pageTwo_length');
      expect(JSON.parse(strJson4).$attrs.content).assertEqual('3');
      console.info('obj4_Router3' + JSON.parse(strJson4).$attrs.content)
      await CommonFunc.sleep(500);


      await CommonFunc.sleep(500);
      let btn1 = await driver.findComponent(ON.id('skip_back'));
      await btn1.click();
      await CommonFunc.sleep(1000);

      let strJson5 = getInspectorByKey('nameState');
      expect(JSON.parse(strJson5).$attrs.content).assertEqual('pageOne');
      console.info('obj5_Router3' + JSON.parse(strJson5).$attrs.content)
      await CommonFunc.sleep(500);

      let strJson6 = getInspectorByKey('getNum');
      expect(JSON.parse(strJson6).$attrs.content).assertEqual('2');
      console.info('obj6_Router3' + JSON.parse(strJson6).$attrs.content)
      await CommonFunc.sleep(100);
      done();
    });
  })
}
