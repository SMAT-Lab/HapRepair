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

export default function CustomLayoutGeometryInfoTest() {
  describe('CustomLayoutGeometryInfoTest', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
     let options:router.RouterOptions = {
        url:'MainAbility/pages/CustomComponentLayout/CustomLayoutGeometryInfoPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get CustomLayoutGeometryInfo state success " + JSON.stringify(pages));
        if (!("CustomLayoutGeometryInfoPage" == pages.name)) {
          console.info("get CustomLayoutGeometryInfo state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push CustomLayoutGeometryInfo page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push CustomLayoutGeometryInfo page error: " + err);
      }
    });

    afterAll(async () => {
      await CommonFunc.sleep(1000);
      if (AppStorage.get('CustomLayoutGeometryInfoPage_margin') != undefined) {
        AppStorage.delete('CustomLayoutGeometryInfoPage_margin');
      }
      if (AppStorage.get('CustomLayoutGeometryInfoPage_padding') != undefined) {
        AppStorage.delete('CustomLayoutGeometryInfoPage_padding');
      }
      console.info("CustomLayoutLayoutChildTest after each called");
    });

    /*
     * @tc.number    CustomLayoutGeometryInfoTest_0010
     * @tc.name      CustomLayoutGeometryInfoTest_0010
     * @tc.desc      test GeometryInfo margin & padding
     * @tc.level     Level 2
     */
    it('CustomLayoutGeometryInfoTest_0010', Level.LEVEL2, async (done: Function) => {
      console.info('[CustomLayoutGeometryInfoTest_0010] START');
      await CommonFunc.sleep(1000);

      console.log('CustomLayoutGeometryInfoTest_0010 margin: ' + JSON.stringify(AppStorage.get('CustomLayoutGeometryInfoPage_margin')));
      console.log('CustomLayoutGeometryInfoTest_0010 padding: ' + JSON.stringify(AppStorage.get('CustomLayoutGeometryInfoPage_padding')));

      expect(JSON.stringify(AppStorage.get('CustomLayoutGeometryInfoPage_margin'))).assertEqual('{"top":5,"right":5,"bottom":5,"left":5}');
      expect(JSON.stringify(AppStorage.get('CustomLayoutGeometryInfoPage_padding'))).assertEqual('{"top":10,"right":10,"bottom":10,"left":10}');

      console.info('[CustomLayoutGeometryInfoTest_0010] END');
      done();
    });
  })
}
