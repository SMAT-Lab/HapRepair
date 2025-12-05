/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
import { expect, it, afterEach, afterAll, beforeEach, beforeAll, describe, Level } from '@ohos/hypium';
import router from '@ohos.router';
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
import CommonFunc from '../../MainAbility/common/Common';
import componentUtils from '@ohos.arkui.componentUtils';

export default function comSizeAbilityTest_10() {
  describe('comSizeAbilityTest_10', () => {
    beforeEach(async (done: Function) => {
      console.info("comSizeAbilityTest_10 beforeEach start")
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/comSizeAbility/comSizeAbilityTest_10'
      }
      try {
        router.clear()
        let pages = router.getState()
        console.info("get comSizeAbilityTest_10 state pages: " + JSON.stringify(pages))
        if (!("comSizeAbilityTest" == pages.name)) {
          console.info("get comSizeAbilityTest_10 state pages.name: " + JSON.stringify(pages.name))
          let result = await router.pushUrl(options)
          await CommonFunc.sleep(1000);
          console.info("push comSizeAbilityTest_10 success: " + JSON.stringify(result))
        }
      } catch (err) {
        console.error("push comSizeAbilityTest_10 page error: " + err)
        expect().assertFail()
      }
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_26', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_26 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_10_to_1'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Path_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Path_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Path_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_26] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_26] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_26] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_26] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_26] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_26] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_26] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_26] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_26] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(210))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(132))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_26] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_27', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_27 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_10_to_2'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Polygon_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Polygon_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Polygon_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_27] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_27] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_27] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_27] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_27] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_27] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_27] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_27] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_27] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(210))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(140))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_27] END')
      await CommonFunc.sleep(1000);
      done()
    })
    /* *
        * @tc.number: 测试编号
        * @tc.name  : comSizeAbilityTestJsunit_01
        * @tc.desc  : test the size of component
        * @tc.level : Level 2
        */
    it('comSizeAbilityTestJsunit_28', Level.LEVEL0, async (done: Function) => {
      console.info('comSizeAbilityTestJsunit_28 START')
      await CommonFunc.sleep(1000);
      let driver = Driver.create();
      let btn = await driver.findComponent(ON.id('comSizeAbilityTest_10_to_3'));
      await btn.click()
      await CommonFunc.sleep(1000);
      let modePosition: componentUtils.ComponentInfo = componentUtils.getRectangleById("Polyline_Size");
      let modePosition1: componentUtils.ComponentInfo = componentUtils.getRectangleById("Polyline_Father");
      let Size = modePosition.size
      let SizeWidth = Size.width
      let SizeHeight = Size.height
      let LocalOffset = modePosition1.localOffset
      let WindowOffset = modePosition.windowOffset
      let ScreenOffset = modePosition.screenOffset
      let Translate = modePosition.translate
      let Transform = modePosition.transform
      let Scale = modePosition.scale
      let Rotate = modePosition.rotate
      let button = await driver.findComponent(ON.id('Polyline_Size'));
      let rect = await button.getBounds();
      console.info("[comSizeAbilityTestJsunit_28] component rect is: " + JSON.stringify(rect))
      console.info("[comSizeAbilityTestJsunit_28] component Size is: " + JSON.stringify(Size))
      console.info("[comSizeAbilityTestJsunit_28] component LocalOffset is: " + JSON.stringify(LocalOffset))
      console.info("[comSizeAbilityTestJsunit_28] component WindowOffset is: " + JSON.stringify(WindowOffset))
      console.info("[comSizeAbilityTestJsunit_28] component ScreenOffset is: " + JSON.stringify(ScreenOffset))
      console.info("[comSizeAbilityTestJsunit_28] component Translate is: " + JSON.stringify(Translate))
      console.info("[comSizeAbilityTestJsunit_28] component Transform is: " + JSON.stringify(Transform))
      console.info("[comSizeAbilityTestJsunit_28] component Scale is: " + JSON.stringify(Scale))
      console.info("[comSizeAbilityTestJsunit_28] component Rotate is: " + JSON.stringify(Rotate))
      await CommonFunc.sleep(1000);
      expect(Math.abs(SizeWidth - Math.round(vp2px(210))) <= 1).assertTrue();
      expect(Math.abs(SizeHeight - Math.round(vp2px(125))) <= 1).assertTrue();
      expect(JSON.stringify(Translate)).assertEqual('{"x":1,"y":1,"z":1}');
      expect(JSON.stringify(Scale)).assertEqual('{"x":1,"y":1,"z":1,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Rotate)).assertEqual('{"x":1,"y":1,"z":1,"angle":0,"centerX":1,"centerY":1}');
      expect(JSON.stringify(Transform)).assertEqual('[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]');
      console.info('[comSizeAbilityTestJsunit_28] END')
      await CommonFunc.sleep(1000);
      done()
    })
  })
}