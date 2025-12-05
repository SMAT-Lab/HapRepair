/**
 * Copyright (c) 2024 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll,  it, Level, expect } from '@ohos/hypium'
import CommonFunc from '../../MainAbility/common/Common';
import { Driver, ON } from '@ohos.UiTest';

export default function PageInfo1() {
  describe('PageInfo1', () => {
    beforeEach(async (done: Function) => {
      console.info('PageInfo1 beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/UIContext/PageInfo1',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get PageInfo1 state success ' + JSON.stringify(pages));
        if (!('PageInfo1' == pages.name)) {
          console.info('get PageInfo1 state success ' + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info('push PageInfo1 page success ' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push PageInfo1 page error: ' + err);
      }
      done()
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('PageInfo1 after each called');
    });
    /**
     * @tc.number    PageInfo1_0100
     * @tc.name      PageInfo1_0100
     * @tc.desc      Test PageInfo
     * @tc.level     Level 2
     */
    it('PageInfo1_0100', Level.LEVEL0, async (done: Function) => {
      console.info('[PageInfo1_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(1000);
      let btn1 = await driver.findComponent(ON.id('PageInfo1_pageOne'));
      await btn1.click()
      await CommonFunc.sleep(1000);
      console.info('[PageInfo1_0100] btn1 click');
      let btn2 = await driver.findComponent(ON.id('PageInfo1_get'));
      await btn2.click()
      await CommonFunc.sleep(1000);
      console.info('[PageInfo1_0100] btn2 click');
      let strJson1 = getInspectorByKey('PageInfo1_text1');
      await CommonFunc.sleep(1000);
      console.info('[PageInfo1_0100] obj1 is: ' + JSON.stringify(JSON.parse(strJson1)));
      expect(JSON.parse(strJson1).$attrs.content).assertEqual('MainAbility/pages/UIContext/PageInfo1');
      let strJson2 = getInspectorByKey('PageInfo1_text2');
      await CommonFunc.sleep(1000);
      console.info('[PageInfo1_0100] obj2 is: ' + JSON.stringify(JSON.parse(strJson2)));
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('pageOne');
      console.info('[PageInfo1_0100] END');
      done();
    });
  })
}