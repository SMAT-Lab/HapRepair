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
import { describe, beforeEach, afterEach, it, expect, Level } from '@ohos/hypium';
import { Driver, ON } from '@ohos.UiTest';
import router from '@ohos.router';
import events_emitter from '@ohos.events.emitter';
import CommonFunc from '../../MainAbility/common/Common';

export default function ListFlingCallback() {
  describe('ListFlingCallback', () => {
    beforeEach(async (done: Function) => {
      console.info('text beforeEach start');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/ListFlingCallback',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info('get ListFlingCallback state pages:' + JSON.stringify(pages));
        if (!('ListFlingCallback' == pages.name)) {
          console.info('get ListFlingCallback state pages.name:' + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info('push ListFlingCallback page result:' + JSON.stringify(result));
        }
      } catch (err) {
        console.error('push ListFlingCallback page error:' + JSON.stringify(err));
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info('ListFlingCallback after each called');
    });

    /**
     * @tc.number ListFlingCallbackTest_0100
     * @tc.name Test the properties of components
     * @tc.desc Set attribute values
     * @tc.size MediumTest
     * @tc.type Function
     * @tc.level Level 1
     */
    it('ListFlingCallbackTest_0100', Level.LEVEL1, async (done: Function) => {
      console.info('ListFlingCallbackTest_0100 START');
      let driver1 = Driver.create();
      let button = await driver1.findComponent(ON.id('btn_list_fling_01'));
      await button.click();
      await CommonFunc.sleep(1000);
      let text1 = getInspectorByKey('key_list_fling_text1');
      let obj: ESObject = JSON.parse(text1);
      expect(obj.$attrs.content).assertEqual('1000');
      console.info('ListFlingCallbackTest_0100 END');
      done();
    });

    /**
     * @tc.number ListFlingCallbackTest_0200
     * @tc.name Test the properties of components
     * @tc.desc Set attribute values
     * @tc.size MediumTest
     * @tc.type Function
     * @tc.level Level 1
     */
    it('ListFlingCallbackTest_0200', Level.LEVEL1, async (done: Function) => {
      console.info('ListFlingCallbackTest_0200 START');
      let driver1 = Driver.create();
      let button = await driver1.findComponent(ON.id('btn_list_fling_02'));
      await button.click();
      await CommonFunc.sleep(1000);
      let text2 = getInspectorByKey('key_list_fling_text2');
      let obj: ESObject = JSON.parse(text2);
      expect(obj.$attrs.content).assertEqual('-1000');
      console.info('ListFlingCallbackTest_0200 END');
      done();
    });
  })
}