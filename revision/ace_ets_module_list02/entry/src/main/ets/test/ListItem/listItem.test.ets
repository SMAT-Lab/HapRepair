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

import events_emitter from '@ohos.events.emitter';
import router from '@ohos.router';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import Utils from '../../MainAbility/common/Utils'

export default function ItemOnSelectJsunit() {
  describe('ItemOnSelectJsunit', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ListItem/list_item',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get list_item state success " + JSON.stringify(pages));
        if (!("list_item" == pages.name)) {
          console.info("get list_item state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await Utils.sleep(2000);
          console.info("push list_item page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push list_item page error: " + err);
      }
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("list_itemOnSelect after each called");
    });

    /*
     * @tc.number    testlist_itemOnSelect0001
     * @tc.name      testlist_itemOnSelect0001
     * @tc.desc      acelist_itemOnSelectEtsTest0001
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testlist_itemOnSelect0001', Level.LEVEL1, async () => {
      console.info('selectOnSelect testlist_itemOnSelect0001 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('ListItem_select');
      console.info("[testlist_itemOnSelect0001] component border strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('ListItem');
      expect(JSON.parse(strJson).$attrs.editable).assertTrue();
      console.info("[testlist_itemOnSelect0001] editable value :" + JSON.parse(strJson).$attrs.editable);

    });

    /*
     * @tc.number    testlist_itemOnSelect0002
     * @tc.name      testlist_itemOnSelect0002
     * @tc.desc      acelist_itemOnSelectEtsTest0002
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testlist_itemOnSelect0002', Level.LEVEL1, async () => {
      console.info('selectOnSelect testlist_itemOnSelect0002 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('ListItem_select');
      console.info("[testlist_itemOnSelect0002] component border strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('ListItem');
      expect(JSON.parse(strJson).$attrs.selectable).assertFalse();
      console.info("[testlist_itemOnSelect0002] selectable value :" + JSON.parse(strJson).$attrs.selectable);

    });

    /*
     * @tc.number    testlist_itemOnSelect0003
     * @tc.name      testlist_itemOnSelect0003
     * @tc.desc      acelist_itemOnSelectEtsTest0002
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testlist_itemOnSelect0003', Level.LEVEL1, async () => {
      console.info('selectOnSelect testlist_itemOnSelect0003 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('ListItem_select');
      console.info("[testlist_itemOnSelect0003] component border strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('ListItem');
      expect(JSON.parse(strJson).$attrs.sticky).assertEqual("Sticky.None");
      console.info("[testlist_itemOnSelect0003] sticky value :" + JSON.parse(strJson).$attrs.sticky);

    });

    /*
     * @tc.number    testlist_itemSwipeAction0004
     * @tc.name      testlist_itemSwipeAction0004
     * @tc.desc      acelist_itemSwipeActionEtsTest0004
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testlist_itemSwipeAction0004', Level.LEVEL1, async () => {
      console.info('selectOnSelect testlist_itemSwipeAction0004 START');
      await Utils.sleep(2000);
      let obj = JSON.stringify(sendEventByKey('Del', 10, ""))
      expect(obj).assertEqual('true');

    });
  })
}
