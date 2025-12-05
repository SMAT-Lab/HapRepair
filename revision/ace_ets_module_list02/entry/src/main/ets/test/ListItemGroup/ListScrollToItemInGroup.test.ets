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
import { describe, beforeEach, afterEach, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router'
import { UiDriver, UiComponent, BY, Driver, Component, ON } from '@ohos.UiTest'
import CommonFunc from '../../MainAbility/common/Common';
import { BusinessError } from '@ohos.base'

export default function ListScrollToItemInGroup() {
  const SUITE: string = 'ListScrollToItemInGroup'
  const waitUiReadyMs = 2000;

  describe('ListScrollToItemInGroup', () => {
    beforeEach(async (done: Function) => {
      let options: router.RouterOptions = {
        url: "MainAbility/pages/ListItemGroup/ListScrollToItemInGroup",
      }
      try {
        router.clear();
        let pages: router.RouterState = router.getState();
        if (pages == null || !("ListScrollToItemInGroup" == pages.name)) {
          await router.pushUrl(options).then(() => {
            console.info(`${SUITE} router.pushUrl success`);
          }).catch((err: BusinessError) => {
            console.error(`${SUITE} router.pushUrl failed, code is ${err.code}, message is ${err.message}`);
          })
        }
      } catch (err) {
        console.error(`${SUITE} beforeEach error:` + JSON.stringify(err));
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(waitUiReadyMs);
      console.info("ListScrollToItemInGroup after each called");
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0001
     * @tc.name      : test initial state
     * @tc.desc      : expect ListItem_0_0 at top position
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0001', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0001] START');
      let List = CommonFunc.getComponentRect('MyList1');
      let ListItem = CommonFunc.getComponentRect('ListItem_Group0_Index0');
      expect(Math.floor(Math.abs(ListItem.top - List.top)) <= 1).assertTrue();
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0001] END');
      done();
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0100
     * @tc.name      : test scrollToItemInGroup inteface
     * @tc.desc      : ListItemGroup0 set ScrollAlign.START, test scrollToItemInGroup in ScrollAlign.START,
     *   expect target ListItem at top position
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0100', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0100] START');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_0_1'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let List = CommonFunc.getComponentRect('MyList1');
      let ListItem = CommonFunc.getComponentRect('ListItem_Group0_Index1');
      expect(Math.floor(Math.abs(ListItem.top - List.top)) <= 1).assertTrue();
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0100] END');
      done();
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0200
     * @tc.name      : test scrollToItemInGroup inteface
     * @tc.desc      ：ListItemGroup1 set ScrollAlign.CENTER,
     *   test scrollToItemInGroup in ScrollAlign.CENTER, expect the target ListItem at center position.
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0200', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0200] START');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_1_2'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let center = CommonFunc.getComponentRect('CenterLine');
      console.info("center.top: " + center.top);
      let ListItem = CommonFunc.getComponentRect('ListItem_Group1_Index2');
      let middle = (ListItem.top + ListItem.bottom) / 2 - vp2px(0.5);
      console.info("ListItem.top: " + ListItem.top + " " + "ListItem.bottom: " + ListItem.bottom);
      console.info("middle: " + middle);
      expect(Math.floor(Math.abs(center.top - middle)) <= 1).assertTrue();
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0200] END');
      done();
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0300
     * @tc.name      : test scrollToItemInGroup inteface
     * @tc.desc      : ListItemGroup1 set ScrollAlign.END,
     *   test scrollToItemInGroup in ScrollAlign.END, expect the target ListItem at bottom position.
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0300', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0300] START');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_2_2'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let List = CommonFunc.getComponentRect('MyList1');
      let ListItem = CommonFunc.getComponentRect('ListItem_Group2_Index2');
      console.info("ListItem.top: " + ListItem.top + " " + "ListItem.bottom: " + ListItem.bottom);
      expect(Math.floor(Math.abs(List.bottom - ListItem.bottom)) <= 1).assertTrue();
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0300] END');
      done();
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0400
     * @tc.name      : test scrollToItemInGroup inteface
     * @tc.desc      : test scrollToItemInGroup in ScrollAlign.AUTO, expect the target ListItem at bottom position
     *   when the distance to the top of List is greater than to bottom.
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0400', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0400] START');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_3_1'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let List = CommonFunc.getComponentRect('MyList1');
      let ListItem = CommonFunc.getComponentRect('ListItem_Group3_Index1');
      expect(Math.floor(Math.abs(ListItem.bottom - List.bottom)) <= 1).assertTrue();
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0400] END');
      done();
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0500
     * @tc.name      : test scrollToItemInGroup inteface
     * @tc.desc      : test scrollToItemInGroup in ScrollAlign.AUTO, expect the target ListItem at end position
     *   when the distance to the bottom of List is less than to the top.
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0500', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0500] START');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_3_11'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let List = CommonFunc.getComponentRect('MyList1');
      let ListItem = CommonFunc.getComponentRect('ListItem_Group3_Index11');
      expect(Math.floor(Math.abs(ListItem.bottom - List.bottom)) <= 1).assertTrue();
      done();
    });

    /**
     * @tc.number    : ArkUI_List_Scroll_To_Item_In_Group_0600
     * @tc.name      : test scrollToItemInGroup inteface
     * @tc.desc      : illegal group, expect nothing happended.
     * @tc.size      : MEDIUM
     * @tc.type      : Function
     * @tc.level     : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0600', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0600] START');
      let OldListItem = CommonFunc.getComponentRect('ListItem_Group3_Index11');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_Err_Group'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let ListItem = CommonFunc.getComponentRect('ListItem_Group3_Index11');
      expect(Math.floor(Math.abs(OldListItem.bottom - ListItem.bottom)) <= 1).assertTrue();
      done();
    });

    /**
     * @tc.number     : ArkUI_List_Scroll_To_Item_In_Group_0700
     * @tc.name       : test scrollToItemInGroup inteface
     * @tc.desc       : illegal index, expect nothing happended.
     * @tc.size       : MEDIUM
     * @tc.type       : Function
     * @tc.level      : Level 1
     */
    it('ArkUI_List_Scroll_To_Item_In_Group_0700', Level.LEVEL1, async (done: Function) => {
      console.info('[ArkUI_List_Scroll_To_Item_In_Group_0700] START');
      let OldListItem = CommonFunc.getComponentRect('ListItem_Group3_Index11');
      let driver: Driver = Driver.create();
      let btn: Component = await driver.findComponent(ON.id('Btn_Err_Group'));
      await btn.click();
      await CommonFunc.sleep(2000);
      let ListItem = CommonFunc.getComponentRect('ListItem_Group3_Index11');
      expect(Math.floor(Math.abs(OldListItem.bottom - ListItem.bottom)) <= 1).assertTrue();
      done();
    });
  })
}
