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
import { Driver, Component, UiWindow, ON } from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';

export default function conSizeGroup1() {
  describe('conSizeGroup1', () => {
    beforeEach(async (done: Function) => {
      console.info("conSizeGroup1 beforeEach start");
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/conSizeGroup1",
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get conSizeGroup1 state pages:" + JSON.stringify(pages));
        if (!("conSizeGroup1" == pages.name)) {
          console.info("get conSizeGroup1 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push conSizeGroup1 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push conSizeGroup1 page error:" + err);
      }
      console.info("conSizeGroup1 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("conSizeGroup1 after each called");
    });

    /**
     * @tc.number    conSizeGroup1_001
     * @tc.name      conSizeGroup1_0100
     * @tc.desc      Set width, height, minWidth, and maxWidth for ListItemGroup.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('conSizeGroup1_0100', Level.LEVEL1, async (done: Function) => {
      console.info('[conSizeGroup1_0100] START');
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('group_constraintSize_1');
      console.info("[conSizeGroup1_0100] component obj is: " + JSON.stringify(JSON.parse(strJson)));
      console.info("[conSizeGroup1_0100] width: " + JSON.stringify(JSON.parse(strJson).$attrs.width));
      await CommonFunc.sleep(1000);
      expect(JSON.parse(strJson).$attrs.width).assertEqual('200.00vp');
      console.info('[conSizeGroup1_0100] END');
      done();
    });
  })
}
