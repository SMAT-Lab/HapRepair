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


export default function Router1() {
  describe('Router1', () => {
    beforeEach(async (done: Function) => {
      console.info("GridFrictionPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/Router1',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get " +
          " state pages:" + JSON.stringify(pages));
        if (!("Router1" == pages.name)) {
          console.info("get Router1 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push Router1 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Router1 page error:" + err);
      }
      console.info("Router1 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back()
      console.info("Router1 after each called");
    });

    /**
     * @tc.number Router1_0100
     * @tc.name Router1_0100
     * @tc.desc Page carrying parameter redirection
     * @tc.level: Level 2
     * @tc.type :
     * @tc.size :
     */
    it('Router1_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[Router1_0100] START');
      console.info('[scrollToJsunit_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(500);
      let btn = await driver.findComponent(ON.id('router1Button'));
      await btn.click();
      await CommonFunc.sleep(2000);


      let strJson1 = getInspectorByKey('data1String');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('message');
      console.info('data1String' + JSON.parse(strJson1).$attrs.content)

      let strJson2 = getInspectorByKey('data2String');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('12345');
      console.info('data2String' + JSON.parse(strJson1).$attrs.content)

      done();
    });
  })
}
