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
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from "@ohos/hypium"
import router from '@ohos.router';
import {UiComponent, Driver, Component, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import CommonFunc from '../../MainAbility/common/Common';
import {MessageManager,Callback} from '../../MainAbility/common/MessageManager';

export default function listVisibleJsunit() {
  describe('listVisibleJsunit', () => {
    beforeEach(async (done: Function) => {
      console.info("listVisibleJsunit beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ListItemGroup/listVisible',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listVisibleJsunit state pages: " + JSON.stringify(pages));
        if (!("listVisible" == pages.name)) {
          console.info("get listVisibleJsunit state pages.name: " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push listVisibleJsunit success: " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listVisibleJsunit page error: " + err);
        expect().assertFail();
      }
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info(" listVisibleJsunit after each called");
    });

    /* *
    * @tc.number: SUB_ACE_SWIPER_ListItemGroupArea_0100
    * @tc.name  : listVisibleJsunit_0100
    * @tc.desc  : Trigger onScrollVisibleContentChange callback.
    * @tc.level : Level 2
    * @tc.type :
    * @tc.size :
    */
    it('listVisibleJsunit_0100', Level.LEVEL2, async (done: Function) => {
      console.info('listVisibleJsunit_0100 START');
      await CommonFunc.sleep(1000);

      let strJson = getInspectorByKey('listVisible_mess');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[listVisibleJsunit_0100] component obj is: " + JSON.stringify(obj));
      expect(obj.$attrs.content).assertEqual('ListItemGroupArea.IN_HEADER_AREA');//IN_HEADER_AREA
      console.info("[listVisibleJsunit_0100] IN_HEADER_AREA success");

      let driver = await Driver.create();
      await CommonFunc.sleep(1000);
      let button1 = await driver.findComponent(ON.id('listVisible_1'));
      await button1.click();
      await CommonFunc.sleep(1000);
      let strJson1 = getInspectorByKey('listVisible_mess');
      let strJson2 = getInspectorByKey('itemVisible_mess');
      let obj1: ESObject = JSON.parse(strJson1);
      let obj2: ESObject = JSON.parse(strJson2);
      console.info("[listVisibleJsunit_0100] component obj1 is: " + JSON.stringify(obj1));
      console.info("[listVisibleJsunit_0100] component obj2 is: " + JSON.stringify(obj2));
      expect(obj1.$attrs.content).assertEqual('ListItemGroupArea.IN_LIST_ITEM_AREA');//IN_LIST_ITEM_AREA
      expect(obj2.$attrs.content).assertEqual('itemIndexInGroup: 0');
      console.info("[listVisibleJsunit_0100] IN_LIST_ITEM_AREA success");


      let button2 = await driver.findComponent(ON.id('listVisible_2'));
      await button2.click();
      await CommonFunc.sleep(1000);
      let strJson3 = getInspectorByKey('listVisible_mess');
      let obj3: ESObject = JSON.parse(strJson3);
      console.info("[listVisibleJsunit_0100] component obj3 is: " + JSON.stringify(obj3));
      expect(obj3.$attrs.content).assertEqual('ListItemGroupArea.IN_FOOTER_AREA');//IN_FOOTER_AREA
      console.info("[listVisibleJsunit_0100] IN_FOOTER_AREA success");

      console.info('[listVisibleJsunit_0100] END');
      done();
    });

  })
}

