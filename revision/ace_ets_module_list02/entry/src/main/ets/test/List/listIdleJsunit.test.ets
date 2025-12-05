/**
 * Copyright (c) 2022 Shenzhen Kaihong Digital Industry Development Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import Utils from '../../MainAbility/common/Utils.ets'
import events_emitter from '@ohos.events.emitter';

export default function listIdleJsunit() {
  describe('listIdleTest', () => {
    beforeEach(async (done: Function) => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/listIdle',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listtest state success " + JSON.stringify(pages));
        if (!("listIdle" == pages.name)) {
          console.info("get listtestI state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await Utils.sleep(2000);
          console.info("push listtest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listtest page error: " + err);
      }
      done()
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("listtestIdle after each called");
    });

    /*
     * @tc.number     testlisttestIdle0001
     * @tc.name       testlisttestIdle0001
     * @tc.desc       acelisttestIdleEtsTest0001
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testlisttestIdle0001', Level.LEVEL1, async (done: Function) => {
      console.info('listtestIdle testlisttestIdle0001 START');
      await Utils.sleep(2000);
      let strJson: string = getInspectorByKey('ScrollList');
      console.info("[testlisttestIdle0001] component controlButton strJson:" + strJson);
      expect(JSON.parse(strJson).$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.editMode).assertEqual(true);
      console.info("[testlisttestIdle0001] editMode value :" + JSON.parse(strJson).$attrs.editMode);
      done();
    });
  })
}
