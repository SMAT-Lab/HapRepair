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
import CommonFunc from '../../MainAbility/common/Common';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function listSContentTest() {
  describe('listSContentTest', () => {
    beforeEach(async (done: Function) => {
      console.info("listSContent beforeEach start");
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listSContent state pages:" + JSON.stringify(pages));
        if (!("listSContent" == pages.name)) {
          console.info("get listSContent pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl({url:"MainAbility/pages/List/listSContent"})
          await CommonFunc.sleep(2000);
          console.info("push listSContent  page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listSContent page error:" + err);
      }
      console.info("listSContent beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listSContent after each called");
    });

    /**
     * @tc.number    SUB_ACE_listSContentTest_0100
     * @tc.name      listSContentTest_0100
     * @tc.desc      test contentStartOffset
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listSContentTest_0100', Level.LEVEL1, async (done: Function) => {
      console.info('[listSContentTest_0100] START');
      await CommonFunc.sleep(2000);
      let json = getInspectorByKey('listTestCon');
      let Obj1: ESObject = JSON.parse(json);
      console.info("listTestCon json info " + JSON.stringify(json));
      expect(Obj1.$attrs.contentStartOffset).assertEqual('10.000000');
      expect(Obj1.$attrs.contentEndOffset).assertEqual('10.000000');
      console.info('[listSContentTest_0100] END');
      done();
    });

  })
}
