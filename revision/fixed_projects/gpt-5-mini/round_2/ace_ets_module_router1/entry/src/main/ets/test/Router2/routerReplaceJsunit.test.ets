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
import {
  UiComponent,
  UiDriver,
  Component,
  Driver,
  UiWindow,
  ON,
  BY,
  MatchPattern,
  DisplayRotation,
  ResizeDirection,
  WindowMode,
  PointerMatrix
} from '@ohos.UiTest';

export default function routerReplaceJsunit() {
  describe('routerReplaceJsunit', () => {
    beforeEach(async (done: Function) => {
      console.info("routerReplaceJsunit beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/routerReplace',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get routerReplaceJsunit state pages:" + JSON.stringify(pages));
        if (!("routerReplace" == pages.name)) {
          console.info("get routerReplaceJsunit pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push routerReplaceJsunit page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push routerReplaceJsunit page error:" + err);
      }
      console.info("routerReplaceJsunit beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("routerReplaceJsunit after each called");
    });

    /**
     * @tc.number routerReplaceJsunit_0100
     * @tc.name routerReplaceJsunit_0100
     * @tc.desc The replace interface does not set params
     * @tc.level Level 3
     * @tc.type
     * @tc.size
     */
    it('routerReplaceJsunit_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[routerReplaceJsunit_0100] START');
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      await CommonFunc.sleep(1000);
      let btn = await driver.findComponent(ON.id('router_replaceUrl_btn'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let text = getInspectorByKey('replace_getParams');
      expect(JSON.parse(text).$attrs.content).assertEqual('router_replaceUrl:undefined');
      console.info('[routerReplaceJsunit_0100] END');
      done();
    });

  })
}