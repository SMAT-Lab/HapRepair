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

export default function listNewJsunit() {
  describe('listNewJsunit', () => {
    beforeAll(async () => {
      console.info("flex beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/listNew',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get list state success " + JSON.stringify(pages));
        if (!("listNew" == pages.name)) {
          console.info("get list state success " + JSON.stringify(pages.name));
          let result = await router.push(options);
          await Utils.sleep(2000);
          console.info("push list page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push list page error: " + err);
      }
    });

    afterEach(async () => {
      await Utils.sleep(1000);
      console.info("listNe after each called");
    });

    /*
     * @tc.number     testListNew0001
     * @tc.name       testListNew0001
     * @tc.desc       acelistNewEtsTest0001
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNew0001', Level.LEVEL1, async () => {
      console.info('listNe testListNew0001 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('list_new1');
      console.info("[testListNew0001] component width strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.width).assertEqual("90.00%");
      console.info("[testListNew0001] width value :" + JSON.parse(strJson).$attrs.width);

    });

    /*
     * @tc.number     testListNe0002
     * @tc.name       testListNe0002
     * @tc.desc       acelistNeEtsTest0002
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNe0002', Level.LEVEL1, async () => {
      console.info('listNe testListNe0002 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('list_new1');
      console.info("[testListNe0002] component height strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.height).assertEqual("300.00vp");
      console.info("[testListNe0002] height value :" + JSON.parse(strJson).$attrs.height);

    });

    /*
     * @tc.number     testListNe0003
     * @tc.name       testListNe0003
     * @tc.desc       acelistNeEtsTest0003
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNe0003', Level.LEVEL1, async () => {
      console.info('listNe testListNe0003 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('list_new1');
      console.info("[testListNe0003] component editMode strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.editMode).assertTrue();
      console.info("[testListNe0003] editMode value :" + JSON.parse(strJson).$attrs.editMode);

    });

    /*
     * @tc.number     testListNe0004
     * @tc.name       testListNe0004
     * @tc.desc       acelistNeEtsTest0004
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNe0004', Level.LEVEL1, async () => {
      console.info('listNe testListNe0004 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('list_new1');
      console.info("[testListNe0004] component alignListItem strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.alignListItem).assertEqual("ListItemAlign.Start");
      console.info("[testListNe0004] alignListItem value :" + JSON.parse(strJson).$attrs.alignListItem);

    });

    /*
     * @tc.number     testListNe0005
     * @tc.name       testListNe0005
     * @tc.desc       acelistNeEtsTest0005
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNe0005', Level.LEVEL1, async () => {
      console.info('listNe testListNe0005 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('list_new1');
      console.info("[testListNe0005] component lanes strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.lanes).assertEqual('4');
      console.info("[testListNe0005] lanes value :" + JSON.parse(strJson).$attrs.lanes);

    });

    /*
     * @tc.number     testListNe0006
     * @tc.name       testListNe0006
     * @tc.desc       acelistNeEtsTest0006
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNe0006', Level.LEVEL1, async () => {
      console.info('listNe testListNe0006 START');
      await Utils.sleep(2000);
      let strJson = getInspectorByKey('list_new1');
      console.info("[testListNe0006] component lanes strJson:" + strJson);
      let obj: Record<string, Object> = JSON.parse(strJson);
      expect(obj.$type).assertEqual('List');
      expect(JSON.parse(strJson).$attrs.lanes).assertEqual('4');
      console.info("[testListNe0006] lanes value :" + JSON.parse(strJson).$attrs.lanes);

    });

    /*
     * @tc.number     testListNe0007
     * @tc.name       testListNe0007
     * @tc.desc       acelistNeEtsTest0007
     * @tc.level      Level 1
     * @tc.type
     * @tc.size
     */
    it('testListNe0007', Level.LEVEL1, async () => {
      console.info('listNe testListNe0007 START');
      let innerEvent: events_emitter.InnerEvent = {
        eventId: 60216,
        priority: events_emitter.EventPriority.LOW
      }
      let callback = (eventData: events_emitter.EventData) => {
        try {
          console.info("callback success");
          console.info("testListNe0007 eventData.data?.result result is: " + eventData.data?.result);
          expect(eventData.data?.result).assertEqual("success");
          console.info("testListNe0007 end");
        } catch (err) {
          console.info("testListNe0007 on events_emitter err : " + JSON.stringify(err));
        }
      }
      try {
        console.info("testListNe0007 click result is: " + JSON.stringify(sendEventByKey('onScrollBegin', 10, "")));
        events_emitter.on(innerEvent, callback);
      } catch (err) {
        console.info("testListNe0007 on events_emitter err : " + JSON.stringify(err));
      }

    });
  })
}
