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

import CommonFunc from '../../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function NavigationCustomTitlePage() {
  describe('NavigationCustomTitlePage', () => {
    beforeEach(async (done: Function) => {
      console.info("NavigationCustomTitlePage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/NavigationCustomTitlePage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get NavigationCustomTitlePage state pages:" + JSON.stringify(pages));
        if (!("NavigationCustomTitlePage" == pages.name)) {
          console.info("get NavigationCustomTitlePage pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push NavigationCustomTitlePage page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push NavigationCustomTitlePage page error:" + err);
      }
      console.info("NavigationCustomTitlePage beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("NavigationCustomTitlePage after each called");
    });


    /**
     * @tc.number    NavigationCustomTitlePage_0100
     * @tc.name      NavigationCustomTitlePage_0100
     * @tc.desc      Test NavigationCustomTitlePage
     * @tc.level     Level 0
     */
    it('NavigationCustomTitlePage_0100', Level.LEVEL0, async (done: Function) => {
      console.info('NavigationCustomTitlePage_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let btn1 = await driver.findComponent(ON.id('customTitleId'));
      await btn1.click()
      console.info('[NavigationCustomTitlePage_0100] btn1 click');
      await CommonFunc.sleep(3000);

      let customTitle:number = AppStorage.get('customTitle') as number
      expect(customTitle).assertEqual(14);
      AppStorage.delete('customTitle');

      console.info('NavigationCustomTitlePage_0100 END');
      done();
    });

    /**
     * @tc.number    NavigationCustomTitlePage_0200
     * @tc.name      NavigationCustomTitlePage_0200
     * @tc.desc      Test NavDestinationCustomTitle
     * @tc.level     Level 0
     */
    it('NavigationCustomTitlePage_0200', Level.LEVEL0, async (done: Function) => {
      console.info('NavigationCustomTitlePage_0200 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(1000);

      let push = await driver.findComponent(ON.text('push'));
      await push.click()
      console.info('[NavigationCustomTitlePage_0200] push click');

      let btn1 = await driver.findComponent(ON.id('customNavTitleId'));
      await btn1.click()
      console.info('[NavigationCustomTitlePage_0200] btn1 click');
      await CommonFunc.sleep(5000);

      let customNavTitle:number = AppStorage.get('customNavTitle') as number;
      expect(customNavTitle).assertEqual(15);
      AppStorage.delete('customNavTitle');

      console.info('NavigationCustomTitlePage_0200 END');
      done();
    });

  })
}