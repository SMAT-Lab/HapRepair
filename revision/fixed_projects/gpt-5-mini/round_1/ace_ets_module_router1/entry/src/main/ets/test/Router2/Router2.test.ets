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

export default function Router2() {
  describe('Router2', () => {
    beforeEach(async (done: Function) => {
      console.info("GridFrictionPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/Router3',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get " +
          " state pages:" + JSON.stringify(pages));
        if (!("Router3" == pages.name)) {
          console.info("get Router2 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push Router2 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push Router2 page error:" + err);
      }
      console.info("Router2 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      router.back()
      console.info("Router2 after each called");
    });
    /**
     * @tc.number Router2_0100
     * @tc.name Router2_0100
     * @tc.desc Page carrying parameter redirection
     * @tc.level: Level 2
     * @tc.type :
     * @tc.size :
     */

    it('Router2_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[Router2_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(500);
      let btn = await driver.findComponent(ON.id('router3Button'));
      await btn.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('arrayData1');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('message');
      console.info('arrayData1' + JSON.parse(strJson1).$attrs.content)
      await CommonFunc.sleep(500);
      let strJson2 = getInspectorByKey('arrayData2');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('1');
      console.info('arrayData2' + JSON.parse(strJson2).$attrs.content)
      await CommonFunc.sleep(100);
      done();
    });
  })
}
