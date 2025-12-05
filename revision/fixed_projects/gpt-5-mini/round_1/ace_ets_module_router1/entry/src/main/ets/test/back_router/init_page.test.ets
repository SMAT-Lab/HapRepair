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
  PointerMatrix,
  UiDirection,
  MouseButton,
  UIElementInfo,
  UIEventObserver
} from '@ohos.UiTest';

export default function aboutToAppearRouterTest() {
  describe('aboutToAppearRouterTest', () => {
    beforeEach(async (done: Function) => {
      console.info("aboutToAppearRouterTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/back_router/init_page',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get aboutToAppearRouterTest state pages:" + JSON.stringify(pages));
        if (!("aboutToAppearRouterTest" == pages.name)) {
          console.info("get WaterFlow_WaterFlow pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push aboutToAppearRouterTest page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push aboutToAppearRouterTest page error:" + err);
      }
      console.info("aboutToAppearRouterTest beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("aboutToAppearRouterTest after each called");
    });
    /**
     * @tc.number aboutToAppearRouterTest_0100
     * @tc.name aboutToAppearRouterTest_0100
     * @tc.desc set api
     * @tc.level Level 2
     * @tc.type
     * @tc.size
     */
    it('aboutToAppearRouterTest_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[aboutToAppearRouterTest_0100] START');
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('Router1_A'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('text_amount');
      let obj1: Record<string, ESObject> = JSON.parse(strJson1);
      let strJson = getInspectorByKey('text_page');
      let obj: Record<string, ESObject> = JSON.parse(strJson);
      expect(obj1.$attrs.content).assertEqual("3");
      expect(obj.$attrs.content).assertEqual("3");
      let button2 = await driver.findComponent(ON.id('RouterA1'));
      await button2.click();
      await CommonFunc.sleep(1000);
      //B页面
      let strJson2 = getInspectorByKey('amount1');
      let obj2: Record<string, ESObject> = JSON.parse(strJson2);
      let strJson3 = getInspectorByKey('page1');
      let obj3: Record<string, ESObject> = JSON.parse(strJson3);
      expect(obj2.$attrs.content).assertEqual("4");
      expect(obj3.$attrs.content).assertEqual("4");
      let button3 = await driver.findComponent(ON.id('RouterB1'));
      await button3.click();
      await CommonFunc.sleep(1000);
      //c页面
      let strJson4 = getInspectorByKey('amountC');
      let obj4: Record<string, ESObject> = JSON.parse(strJson4);
      let strJson5 = getInspectorByKey('pageC');
      let obj5: Record<string, ESObject> = JSON.parse(strJson5);
      expect(obj4.$attrs.content).assertEqual("5");
      expect(obj5.$attrs.content).assertEqual("5");
      let button4 = await driver.findComponent(ON.id('RouteC2'));
      await button4.click();
      await CommonFunc.sleep(1000);
      //返回B页
      expect(obj2.$attrs.content).assertEqual("4");
      expect(obj3.$attrs.content).assertEqual("4");

      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let button5 = await scrollBar.scrollSearch(ON.id('RouteB2'));
      await button5.click();
      await CommonFunc.sleep(1000);
      //返回A页
      expect(obj1.$attrs.content).assertEqual("3");
      expect(obj.$attrs.content).assertEqual("3");
      
      let scrollBar2: Component = await driver.findComponent(ON.type('Scroll'));
      let button6 = await scrollBar2.scrollSearch(ON.id('RouterA2'));
      await button6.click();
      await CommonFunc.sleep(1000);
      //返回初始页
      let strJson6 = getInspectorByKey('amountIndex');
      let obj6: Record<string, ESObject> = JSON.parse(strJson6);
      let strJson7 = getInspectorByKey('pageIndex');
      let obj7: Record<string, ESObject> = JSON.parse(strJson7);
      expect(obj6.$attrs.content).assertEqual("2");
      expect(obj7.$attrs.content).assertEqual("2");
      await CommonFunc.sleep(1000);
      done();
    });

    /**
     * @tc.number aboutToAppearRouterTest_0200
     * @tc.name aboutToAppearRouterTest
     * @tc.desc set api
     * @tc.level Level 2
     * @tc.type
     * @tc.size
     */
    it('aboutToAppearRouterTest_0200', Level.LEVEL2, async (done: Function) => {
      console.info('[aboutToAppearRouterTest_0200] START');
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('Router1_A'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('text_amount');
      let strJson = getInspectorByKey('text_page');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual("3");
      expect(JSON.parse(strJson).$attrs.content).assertEqual("3");
      let button2 = await driver.findComponent(ON.id('RouterA1'));
      await button2.click();
      await CommonFunc.sleep(1000);
      let strJson2 = getInspectorByKey('amount1');
      let strJson3 = getInspectorByKey('page1');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual("4");
      expect(JSON.parse(strJson3).$attrs.content).assertEqual("4");
      let button3 = await driver.findComponent(ON.id('RouterB1'));
      await button3.click();
      await CommonFunc.sleep(1000);
      let strJson4 = getInspectorByKey('amountC');
      let strJson5 = getInspectorByKey('pageC');
      expect(JSON.parse(strJson4).$attrs.content).assertEqual("5");
      expect(JSON.parse(strJson5).$attrs.content).assertEqual("5");
      let button4 = await driver.findComponent(ON.id('RouteC2_A'));
      await button4.click();
      await CommonFunc.sleep(1000);
      expect(JSON.parse(strJson4).$attrs.content).assertEqual("5");
      expect(JSON.parse(strJson5).$attrs.content).assertEqual("5");
      await CommonFunc.sleep(1000);
      done();
    });
  })
}
