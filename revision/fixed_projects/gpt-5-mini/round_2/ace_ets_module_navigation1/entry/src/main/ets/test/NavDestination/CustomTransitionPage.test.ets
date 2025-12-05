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

export default function CustomTransitionPage() {
  describe('CustomTransitionPage', () => {
    beforeEach(async (done: Function) => {
      console.info("CustomTransitionPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/NavDestination/CustomTransitionPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get CustomTransitionPage state pages:" + JSON.stringify(pages));
        if (!("CustomTransitionPage" == pages.name)) {
          console.info("get CustomTransitionPage pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push CustomTransitionPage page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push CustomTransitionPage page error:" + err);
      }
      console.info("CustomTransitionPage beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("CustomTransitionPage after each called");
    });


    /**
     * @tc.number    CustomTransitionPage_0100
     * @tc.name      CustomTransitionPage_0100
     * @tc.desc      Test customTransitionPage return undefined, push enter
     * @tc.level     Level 2
     */
    it('CustomTransitionPage_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('CustomTransitionPage_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('pushOneId'));
      await btn1.click()
      console.info('[CustomTransitionPage_0100] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('pushTrueId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[CustomTransitionPage_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[CustomTransitionPage_0100] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('50');

      console.info('CustomTransitionPage_0100 END');
      done();
    });

    /**
     * @tc.number    CustomTransitionPage_0200
     * @tc.name      CustomTransitionPage_0200
     * @tc.desc      Test customTransitionPage return array, push enter and exit
     * @tc.level     Level 2
     */
    it('CustomTransitionPage_0200',  Level.LEVEL0, async (done: Function) => {
      console.info('CustomTransitionPage_0200 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('pushId'));
      await btn1.click()
      console.info('[CustomTransitionPage_0200] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('pushTrueId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[CustomTransitionPage_0200] component obj is: ' + JSON.stringify(obj));
      console.info('[CustomTransitionPage_0200] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('100');

      let strJson2: string = getInspectorByKey('pushFalseId');
      let obj2: Record<string, ESObject> = JSON.parse(strJson2);
      console.info('[CustomTransitionPage_0200] component obj2 is: ' + JSON.stringify(obj2));
      console.info('[CustomTransitionPage_0200] content obj2 is: ' + JSON.stringify(obj2.$attrs.content));
      expect(obj2.$attrs.content).assertEqual('push_0');

      console.info('CustomTransitionPage_0200 END');
      done();
    });


    /**
     * @tc.number    CustomTransitionPage_0300
     * @tc.name      CustomTransitionPage_0300
     * @tc.desc      Test customTransitionPage return array, pop enter and exit
     * @tc.level     Level 2
     */
    it('CustomTransitionPage_0300',  Level.LEVEL0, async (done: Function) => {
      console.info('CustomTransitionPage_0300 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('popId'));
      await btn1.click()
      console.info('[CustomTransitionPage_0300] btn1 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('popTrueId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[CustomTransitionPage_0300] component obj is: ' + JSON.stringify(obj));
      console.info('[CustomTransitionPage_0300] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('pop_1');

      let strJson2: string = getInspectorByKey('popFalseId');
      let obj2: Record<string, ESObject> = JSON.parse(strJson2);
      console.info('[CustomTransitionPage_0200] component obj2 is: ' + JSON.stringify(obj2));
      console.info('[CustomTransitionPage_0200] content obj2 is: ' + JSON.stringify(obj2.$attrs.content));
      expect(obj2.$attrs.content).assertEqual('pop_0');

      console.info('CustomTransitionPage_0300 END');
      done();
    });

    /**
     * @tc.number    CustomTransitionPage_0400
     * @tc.name      CustomTransitionPage_0400
     * @tc.desc      Test customTransitionPage return array, replacePath enter and exit
     * @tc.level     Level 2
     */
    it('CustomTransitionPage_0400',  Level.LEVEL0, async (done: Function) => {
      console.info('CustomTransitionPage_0400 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('pushId'));
      await btn1.click()
      console.info('[CustomTransitionPage_0400] btn1 click');
      await CommonFunc.sleep(500);
      let btn2 = await driver.findComponent(ON.id('replaceId'));
      await btn2.click()
      console.info('[CustomTransitionPage_0400] btn2 click');
      await CommonFunc.sleep(1000);

      let strJson: string = getInspectorByKey('replaceTrueId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[CustomTransitionPage_0400] component obj is: ' + JSON.stringify(obj));
      console.info('[CustomTransitionPage_0400] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('replace_1');

      let strJson2: string = getInspectorByKey('replaceFalseId');
      let obj2: Record<string, ESObject> = JSON.parse(strJson2);
      console.info('[CustomTransitionPage_0400] component obj2 is: ' + JSON.stringify(obj2));
      console.info('[CustomTransitionPage_0400] content obj2 is: ' + JSON.stringify(obj2.$attrs.content));
      expect(obj2.$attrs.content).assertEqual('replace_0');

      console.info('CustomTransitionPage_0400 END');
      done();
    });
  })
}