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

export default function OnResultPage() {
  describe('OnResultPage', () => {
    beforeEach(async (done: Function) => {
      console.info("OnResultPage beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Navigation/OnResultPage',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get OnResultPage state pages:" + JSON.stringify(pages));
        if (!("OnResultPage" == pages.name)) {
          console.info("get OnResultPage pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push OnResultPage page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push OnResultPage page error:" + err);
      }
      console.info("OnResultPage beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("OnResultPage after each called");
    });


    /**
     * @tc.number    OnResultPage_0100
     * @tc.name      OnResultPage_0100
     * @tc.desc      Test OnResult pop
     * @tc.level     Level 2
     */
    it('OnResultPage_0100',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0100 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn1 = await driver.findComponent(ON.id('pushId'));
      await btn1.click()
      console.info('[OnResultPage_0100] btn1 click');
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushTwoId'));
      await btn2.click()
      console.info('[OnResultPage_0100] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('popId'));
      await btn3.click()
      console.info('[OnResultPage_0100] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0100] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0100] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('test');

      console.info('OnResultPage_0100 END');
      done();
    });

    /**
     * @tc.number    OnResultPage_0200
     * @tc.name      OnResultPage_0200
     * @tc.desc      Test OnResult popToIndex
     * @tc.level     Level 2
     */
    it('OnResultPage_0200',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0200 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushTwoId'));
      await btn2.click()
      console.info('[OnResultPage_0200] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('popIndexId'));
      await btn3.click()
      console.info('[OnResultPage_0200] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0200] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0200] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('2');

      console.info('OnResultPage_0200 END');
      done();
    });

    /**
     * @tc.number    OnResultPage_0300
     * @tc.name      OnResultPage_0300
     * @tc.desc      Test OnResult popToName
     * @tc.level     Level 2
     */
    it('OnResultPage_0300',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0300 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushTwoId'));
      await btn2.click()
      console.info('[OnResultPage_0300] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('popNameId'));
      await btn3.click()
      console.info('[OnResultPage_0300] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0300] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0300] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('popToName');

      console.info('OnResultPage_0300 END');
      done();
    });

    /**
     * @tc.number    OnResultPage_0400
     * @tc.name      OnResultPage_0400
     * @tc.desc      Test OnResult undefined
     * @tc.level     Level 2
     */
    it('OnResultPage_0400',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0400 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushTwoId'));
      await btn2.click()
      console.info('[OnResultPage_0400] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('popUndefinedId'));
      await btn3.click()
      console.info('[OnResultPage_0400] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0400] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0400] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('undefined');

      console.info('OnResultPage_0400 END');
      done();
    });

    /**
     * @tc.number    OnResultPage_0500
     * @tc.name      OnResultPage_0500
     * @tc.desc      Test OnResult popToName
     * @tc.level     Level 2
     */
    it('OnResultPage_0500',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0500 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushTwoId'));
      await btn2.click()
      console.info('[OnResultPage_0500] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('popNullId'));
      await btn3.click()
      console.info('[OnResultPage_0500] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0500] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0500] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('null');

      console.info('OnResultPage_0500 END');
      done();
    });


    /**
     * @tc.number    OnResultPage_0600
     * @tc.name      OnResultPage_0600
     * @tc.desc      Test push dialog page pop
     * @tc.level     Level 2
     */
    it('OnResultPage_0600',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0600 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushDialogId'));
      await btn2.click()
      console.info('[OnResultPage_0600] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('dialogPopId'));
      await btn3.click()
      console.info('[OnResultPage_0600] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0600] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0600] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('test');

      console.info('OnResultPage_0600 END');
      done();
    });

    /**
     * @tc.number    OnResultPage_0700
     * @tc.name      OnResultPage_0700
     * @tc.desc      Test push dialog page popToIndex
     * @tc.level     Level 2
     */
    it('OnResultPage_0700',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0700 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushDialogId'));
      await btn2.click()
      console.info('[OnResultPage_0700] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('dialogPopIndexId'));
      await btn3.click()
      console.info('[OnResultPage_0700] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0700] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0700] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('popToIndex');

      console.info('OnResultPage_0700 END');
      done();
    });

    /**
     * @tc.number    OnResultPage_0800
     * @tc.name      OnResultPage_0800
     * @tc.desc      Test push dialog page popToName
     * @tc.level     Level 2
     */
    it('OnResultPage_0800',  Level.LEVEL0, async (done: Function) => {
      console.info('OnResultPage_0800 START');
      await CommonFunc.sleep(1000);

      let driver = Driver.create();
      await CommonFunc.sleep(200);

      let btn2 = await driver.findComponent(ON.id('pushDialogId'));
      await btn2.click()
      console.info('[OnResultPage_0800] btn2 click');
      await CommonFunc.sleep(200);

      let btn3 = await driver.findComponent(ON.id('dialogPopNameId'));
      await btn3.click()
      console.info('[OnResultPage_0800] btn3 click');
      await CommonFunc.sleep(500);

      let strJson: string = getInspectorByKey('infoId');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      console.info('[OnResultPage_0800] component obj is: ' + JSON.stringify(obj));
      console.info('[OnResultPage_0800] content obj is: ' + JSON.stringify(obj.$attrs.content));
      expect(obj.$attrs.content).assertEqual('popToName');

      console.info('OnResultPage_0800 END');
      done();
    });
  })
}