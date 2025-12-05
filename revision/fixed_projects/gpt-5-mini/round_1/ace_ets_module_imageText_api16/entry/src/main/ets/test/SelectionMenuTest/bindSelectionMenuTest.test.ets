/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import { Driver, ON, MouseButton } from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common'

export default function bindSelectionMenuTest() {
  describe('bindSelectionMenuTest', () => {
    beforeEach(async (done: Function) => {
      console.info("bindSelectionMenuTest beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/SelectionMenuTest/bindSelectionMenuTest',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get bindSelectionMenuTest state success " + JSON.stringify(pages));
        let result = await router.replaceUrl(options);
        if (!("bindSelectionMenuTest" == pages.name)) {
          console.info("get bindSelectionMenuTest state success " + JSON.stringify(pages.name));
          await CommonFunc.sleep(2000);
          console.info("push bindSelectionMenuTest page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push bindSelectionMenuTest page error: " + err);
      }
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("bindSelectionMenuTest after each called");
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0001
    * @tc.name         bindSelectionMenuTest0001
    * @tc.desic        Test Text by mouseClick
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0001', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0001 START');
        await CommonFunc.sleep(1000);
        let Text = await driver.findComponent(ON.id('text'))
        await CommonFunc.sleep(500);
        let point = await Text.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.mouseClick(point, MouseButton.MOUSE_BUTTON_RIGHT);
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0001] end");
        done();
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0002
    * @tc.name         bindSelectionMenuTest0002
    * @tc.desic        Test Text by longClick
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0002', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0002 START');
        await CommonFunc.sleep(1000);
        let Text = await driver.findComponent(ON.id('text'))
        await CommonFunc.sleep(500);
        await Text.longClick();
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0002] end");
        done();
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0003
    * @tc.name         bindSelectionMenuTest0003
    * @tc.desic        Test  Text by select
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0003', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0003 START');
        await CommonFunc.sleep(1000);
        let Text = await driver.findComponent(ON.id('text'))
        await CommonFunc.sleep(500);
        let point = await Text.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.mouseDrag({x:Math.round(point.x-vp2px(20)), y:Math.round(point.y)},{x:Math.round(point.x+vp2px(20)), y:Math.round(point.y)},600);
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0003] end");
        done();
    });


    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0004
    * @tc.name         bindSelectionMenuTest0004
    * @tc.desic        Test image text by mouseClick
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0004', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0004 START');
        await CommonFunc.sleep(1000);
        let image = await driver.findComponent(ON.id('image'))
        await CommonFunc.sleep(500);
        let point = await image.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.mouseClick(point, MouseButton.MOUSE_BUTTON_RIGHT);
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0004] end");
        done();
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0005
    * @tc.name         bindSelectionMenuTest0005
    * @tc.desic        Test image text by longClick
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0005', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0005 START');
        await CommonFunc.sleep(1000);
        let image = await driver.findComponent(ON.id('image'))
        await CommonFunc.sleep(500);
        await image.longClick();
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0005] end");
        done();
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0006
    * @tc.name         bindSelectionMenuTest0006
    * @tc.desic        Test image Text by select
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0006', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0006 START');
        await CommonFunc.sleep(1000);
        let image = await driver.findComponent(ON.id('image'))
        await CommonFunc.sleep(500);
        let point = await image.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.mouseDrag({x:Math.round(point.x-vp2px(24)), y:Math.round(point.y)},{x:Math.round(point.x+vp2px(24)), y:Math.round(point.y)},600);
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0006] end");
        done();
    });



    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0007
    * @tc.name         bindSelectionMenuTest0007
    * @tc.desic        Test mixed Text by mouseClick
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0007', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0007 START');
        await CommonFunc.sleep(1000);
        let mixed = await driver.findComponent(ON.id('mixed'))
        await CommonFunc.sleep(500);
        let point = await mixed.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.mouseClick(point, MouseButton.MOUSE_BUTTON_RIGHT);
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0007] end");
        done();
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0008
    * @tc.name         bindSelectionMenuTest0008
    * @tc.desic        Test mixed Text by longClick
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0008', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0008 START');
        await CommonFunc.sleep(1000);
        let mixed = await driver.findComponent(ON.id('mixed'))
        await CommonFunc.sleep(500);
        await mixed.longClick();
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0008] end");
        done();
    });

    /*
    * @tc.number       SUB_ACE_BASIC_ETS_API_0009
    * @tc.name         bindSelectionMenuTest0009
    * @tc.desic        Test mixed Text by select
	* @tc.size         MediumTest
	* @tc.type         Function
	* @tc.level        1
     */
    it('bindSelectionMenuTest0009', Level.LEVEL1, async (done: Function) => {
        let driver = Driver.create();
        console.info('bindSelectionMenuTest bindSelectionMenuTest0009 START');
        await CommonFunc.sleep(1000);
        let mixed = await driver.findComponent(ON.id('mixed'))
        await CommonFunc.sleep(500);
        let point = await mixed.getBoundsCenter();
        await CommonFunc.sleep(500);
        await driver.mouseDrag({x:Math.round(point.x-vp2px(20)), y:Math.round(point.y)},{x:Math.round(point.x+vp2px(50)), y:Math.round(point.y)},600);
        await CommonFunc.sleep(500);
        await driver.assertComponentExist(ON.text('自定义菜单'));
        console.info("[bindSelectionMenuTest0009] end");
        done();
    });
  })
}