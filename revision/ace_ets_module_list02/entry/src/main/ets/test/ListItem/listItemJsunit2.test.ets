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
import {describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import events_emitter from '@ohos.events.emitter';
import CommonFunc from '../../MainAbility/common/Common';

export default function listItemJsunit2() {
  describe('listItemJsunit2', () => {
    beforeEach(async (done: Function) => {
      console.info("listItem beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/ListItem/listItem2',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listItem state pages:" + JSON.stringify(pages));
        if (!("listItem2" == pages.name)) {
          console.info("get listItem state pages.name:" + JSON.stringify(pages.name));
          let result = await router.push(options);
          await CommonFunc.sleep(2000);
          console.info("push listItem page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listItem page error:" + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listItem after each called");
    });

    /**
     * @tc.number    testListItem01
     * @tc.name      testListItem01
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testListItem01', Level.LEVEL1, async (done: Function) => {
      console.info('[testListItem01] START');
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('ListItem_2');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[testListItem01] obj is: " + JSON.stringify(obj));
      expect(obj.$attrs.sticky).assertEqual('Sticky.Normal');
      console.info('testListItem01 END');
      done();
    });

    /**
     * @tc.number    testListItem02
     * @tc.name      testListItem02
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testListItem02', Level.LEVEL1, async (done: Function) => {
      console.info('[testListItem02] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "sticky": Sticky.None
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 122,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testListItem02] start to publish emit");
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testListItem02] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('ListItem_2');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[testListItem02] obj is: " + JSON.stringify(obj));
      expect(obj.$attrs.sticky).assertEqual('Sticky.None');
      console.info('testListItem02 END');
      done();
    });

    /**
     * @tc.number    testListItem03
     * @tc.name      testListItem03
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testListItem03', Level.LEVEL1, async (done: Function) => {
      console.info('[testListItem03] START');
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('List_2');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[testListItem03] obj is: " + JSON.stringify(obj));
      expect(obj.$attrs.editMode).assertTrue();
      console.info('testListItem03 END');
      done();
    });

    /**
     * @tc.number    testListItem04
     * @tc.name      testListItem04
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testListItem04', Level.LEVEL1, async (done: Function) => {
      console.info('[testEllipse03] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "editMode": false
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 123,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testListItem04] start to publish emit");
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testListItem04] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson = getInspectorByKey('List_2');
      let obj: ESObject = JSON.parse(strJson);
      console.info("[testListItem04] obj is: " + JSON.stringify(obj));
      expect(obj.$attrs.editMode).assertFalse();
      console.info('testListItem04 END');
      done();
    });
  })
}
