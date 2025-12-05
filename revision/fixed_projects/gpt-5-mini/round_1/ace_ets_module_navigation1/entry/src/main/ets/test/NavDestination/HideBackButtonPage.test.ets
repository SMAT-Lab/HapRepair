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

export default function HideBackButtonPage() {
  describe('HideBackButtonPage', () => {
    beforeEach(async (done: Function) => {
      console.info("HideBackButtonPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/NavDestination/HideBackButtonPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get HideBackButtonPage state pages:" + JSON.stringify(pages));
        if (!("HideBackButtonPage" == pages.name)) {
          console.info("get HideBackButtonPage pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push HideBackButtonPage page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push HideBackButtonPage page error:" + err);
      }
      console.info("HideBackButtonPage beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("HideBackButtonPage after each called");
    });


    /**
     * @tc.number    HideBackButtonPage_0100
     * @tc.name      HideBackButtonPage_0100
     * @tc.desc      Test HideBackButtonPage set hideBackButton is false
     * @tc.level     Level 2
     */
    it('HideBackButtonPage_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('HideBackButtonPage_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('pushOneId'));
      await btn1.click()
      console.info('[CustomTransitionPage_0100] btn1 click');
      await CommonFunc.sleep(500);

      let button: Component | null = await driver.findComponent(ON.type('BackButton'));
      expect(button).assertEqual(null);

      console.info('HideBackButtonPage_0100 END');
      done();
    });

    /**
     * @tc.number    HideBackButtonPage_0200
     * @tc.name      HideBackButtonPage_0200
     * @tc.desc      Test HideBackButtonPage set hideBackButton is true
     * @tc.level     Level 2
     */
    it('HideBackButtonPage_0200',  Level.LEVEL0, async (done: Function) => {
      console.info('HideBackButtonPage_0200 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btnPop = await driver.findComponent(ON.id('popId'));
      await btnPop.click()
      await CommonFunc.sleep(500);

      let btn1 = await driver.findComponent(ON.id('pushTwoId'));
      await btn1.click()
      console.info('[HideBackButtonPage_0200] btn1 click');
      await CommonFunc.sleep(500);

      await driver.assertComponentExist(ON.type('BackButton'));

      let backButton2: Component | null = await driver.findComponent(ON.type('BackButton'));
      expect(backButton2 !== null).assertTrue();

      console.info('HideBackButtonPage_0200 END');
      done();
    });


    /**
     * @tc.number    HideBackButtonPage_0300
     * @tc.name      HideBackButtonPage_0300
     * @tc.desc      Test HideBackButtonPage set hideBackButton is null
     * @tc.level     Level 2
     */
    it('HideBackButtonPage_0300',  Level.LEVEL0, async (done: Function) => {
      console.info('HideBackButtonPage_0300 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btnPop = await driver.findComponent(ON.id('popId'));
      await btnPop.click()
      await CommonFunc.sleep(500);

      let btn1 = await driver.findComponent(ON.id('pageThreeId'));
      await btn1.click()
      console.info('[HideBackButtonPage_0300] btn1 click');
      await CommonFunc.sleep(500);

      await driver.assertComponentExist(ON.type('BackButton'));

      let backButton3: Component | null = await driver.findComponent(ON.type('BackButton'));
      expect(backButton3 !== null).assertTrue();

      console.info('HideBackButtonPage_0300 END');
      done();
    });

    /**
     * @tc.number    HideBackButtonPage_0400
     * @tc.name      HideBackButtonPage_0400
     * @tc.desc      Test HideBackButtonPage set hideBackButton is undefined
     * @tc.level     Level 2
     */
    it('HideBackButtonPage_0400',  Level.LEVEL0, async (done: Function) => {
      console.info('HideBackButtonPage_0400 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btnPop = await driver.findComponent(ON.id('popId'));
      await btnPop.click()
      await CommonFunc.sleep(500);

      let btn1 = await driver.findComponent(ON.id('pushFourId'));
      await btn1.click()
      console.info('[HideBackButtonPage_0300] btn1 click');
      await CommonFunc.sleep(500);

      await driver.assertComponentExist(ON.type('BackButton'));

      let backButton4: Component | null = await driver.findComponent(ON.type('BackButton'));
      expect(backButton4 !== null).assertTrue();

      console.info('HideBackButtonPage_0400 END');
      done();
    });

  })
}