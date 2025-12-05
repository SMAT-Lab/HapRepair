/*
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

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import CommonFunc from '../../MainAbility/common/Common';

export default function CustomLayoutLayoutChildTest() {
  describe('CustomLayoutLayoutChildTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
     let options:router.RouterOptions = {
        url:'MainAbility/pages/CustomComponentLayout/CustomLayoutLayoutChildPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get CustomLayoutLayoutChild state success " + JSON.stringify(pages));
        if (!("CustomLayoutLayoutChildPage" == pages.name)) {
          console.info("get CustomLayoutLayoutChild state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push CustomLayoutLayoutChild page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push CustomLayoutLayoutChild page error: " + err);
      }
    });

    afterAll(async () => {
      await CommonFunc.sleep(1000);
      if (AppStorage.get('CustomLayoutLayoutChildPage_id_0') != undefined) {
        AppStorage.delete('CustomLayoutLayoutChildPage_id_0');
      }
      if (AppStorage.get('CustomLayoutLayoutChildPage_id_1') != undefined) {
        AppStorage.delete('CustomLayoutLayoutChildPage_id_1');
      }
      if (AppStorage.get('CustomLayoutLayoutChildPage_id_2') != undefined) {
        AppStorage.delete('CustomLayoutLayoutChildPage_id_2');
      }
      if (AppStorage.get('CustomLayoutLayoutChildPage_name_0') != undefined) {
        AppStorage.delete('CustomLayoutLayoutChildPage_name_0');
      }
      if (AppStorage.get('CustomLayoutLayoutChildPage_name_1') != undefined) {
        AppStorage.delete('CustomLayoutLayoutChildPage_name_1');
      }
      if (AppStorage.get('CustomLayoutLayoutChildPage_name_2') != undefined) {
        AppStorage.delete('CustomLayoutLayoutChildPage_name_2');
      }
      console.info("CustomLayoutLayoutChildTest after each called");
    });

    /*
     * @tc.number    CustomLayoutLayoutChildTest_0010
     * @tc.name      CustomLayoutLayoutChildTest_0010
     * @tc.desc      test LayoutChild id & name
     * @tc.level     Level 2
     */
    it('CustomLayoutLayoutChildTest_0010', Level.LEVEL2, async (done: Function) => {
      console.info('[CustomLayoutLayoutChildTest_0010] START');
      await CommonFunc.sleep(1000);

      console.info('[CustomLayoutLayoutChildTest_0010] child_0 ID: ' + AppStorage.get('CustomLayoutLayoutChildPage_id_0'));
      console.info('[CustomLayoutLayoutChildTest_0010] child_1 ID: ' + AppStorage.get('CustomLayoutLayoutChildPage_id_1'));
      console.info('[CustomLayoutLayoutChildTest_0010] child_2 ID: ' + AppStorage.get('CustomLayoutLayoutChildPage_id_2'));
      console.info('[CustomLayoutLayoutChildTest_0010] child_0 name: ' + AppStorage.get('CustomLayoutLayoutChildPage_name_0'));
      console.info('[CustomLayoutLayoutChildTest_0010] child_1 name: ' + AppStorage.get('CustomLayoutLayoutChildPage_name_1'));
      console.info('[CustomLayoutLayoutChildTest_0010] child_2 name: ' + AppStorage.get('CustomLayoutLayoutChildPage_name_2'));
      //expect(AppStorage.get('CustomLayoutLayoutChildPage_id_0')).assertEqual('29');
      //expect(AppStorage.get('CustomLayoutLayoutChildPage_id_1')).assertEqual('31');
      //expect(AppStorage.get('CustomLayoutLayoutChildPage_id_2')).assertEqual('33');
      expect(AppStorage.get('CustomLayoutLayoutChildPage_name_0')).assertEqual('Text');
      expect(AppStorage.get('CustomLayoutLayoutChildPage_name_1')).assertEqual('Text');
      expect(AppStorage.get('CustomLayoutLayoutChildPage_name_2')).assertEqual('Text');

      console.info('[CustomLayoutLayoutChildTest_0010] END');
      done();
    });
  })
}
