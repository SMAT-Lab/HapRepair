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

export default function aboutToAppearRouterTest2() {
  describe('aboutToAppearRouterTest2', () => {
    beforeEach(async (done: Function) => {
      console.info("aboutToAppearRouterTest2 beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/back_router/init_page',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get aboutToAppearRouterTest2 state pages:" + JSON.stringify(pages));
        if (!("aboutToAppearRouterTest2" == pages.name)) {
          console.info("get WaterFlow_WaterFlow pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push aboutToAppearRouterTest2 page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push aboutToAppearRouterTest2 page error:" + err);
      }
      console.info("aboutToAppearRouterTest2 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("aboutToAppearRouterTest2 after each called");
    });
    /**
     * @tc.number aboutToAppearRouterTest2_0100
     * @tc.name aboutToAppearRouterTest2_0100
     * @tc.desc set api
     * @tc.level Level 2
     * @tc.type
     * @tc.size
     */
    it('aboutToAppearRouterTest2_0100', Level.LEVEL2, async (done: Function) => {
      console.info('[aboutToAppearRouterTest2_0100] START');
      let driver = Driver.create();
      let button = await driver.findComponent(ON.id('Router1_A'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('text_amount');
      let strJson = getInspectorByKey('text_page');
      expect(JSON.parse(strJson1).$attrs.content).assertEqual("3");
      expect(JSON.parse(strJson).$attrs.content).assertEqual("3");

      let scrollBar: Component = await driver.findComponent(ON.type('Scroll'));
      let button2 = await scrollBar.scrollSearch(ON.id('RouterA1'));
      await button2.click();
      await CommonFunc.sleep(1000);
      //B页面
      let strJson2 = getInspectorByKey('amount1');
      let strJson3 = getInspectorByKey('page1');
      expect(JSON.parse(strJson2).$attrs.content).assertEqual("4");
      expect(JSON.parse(strJson3).$attrs.content).assertEqual("4");
      //清空所有页
      await CommonFunc.sleep(1000);

      let scrollBar2: Component = await driver.findComponent(ON.type('Scroll'));
      let button4 = await scrollBar2.scrollSearch(ON.id('clear'))
      await button4.click();
      await CommonFunc.sleep(3000);
      expect(JSON.parse(getInspectorByKey('index')).$attrs.content).assertEqual("1");
      expect(JSON.parse(getInspectorByKey('length')).$attrs.content).assertEqual("1");
      await CommonFunc.sleep(1000);
      //返回上一页
      let scrollBar3: Component = await driver.findComponent(ON.type('Scroll'));
      let button5 = await scrollBar3.scrollSearch(ON.id('ReturnBack'))
      await button5.click();
      await CommonFunc.sleep(1000);
      expect(JSON.parse(getInspectorByKey('index')).$attrs.content).assertEqual("1");
      expect(JSON.parse(getInspectorByKey('length')).$attrs.content).assertEqual("1");
      await CommonFunc.sleep(1000);
      done();
    });

  })
}
