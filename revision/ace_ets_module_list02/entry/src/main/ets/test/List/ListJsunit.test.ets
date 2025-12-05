
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
import events_emitter from '@ohos.events.emitter';
import CommonFunc from '../../MainAbility/common/Common';

export default function listJsunit() {
  describe('listTest', () => {
    beforeEach(async (done:Function) => {
      console.info("list beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/List/list',
      }
      try {
        router.clear();
        let pages:router.RouterState = router.getState();
        console.info("get list state pages:" + JSON.stringify(pages));
        if (!("list" == pages.name)) {
          console.info("get list state pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push list page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push list page error:" + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("list after each called");
    });

    /**
     * @tc.number    testList01
     * @tc.name      testList01
     * @tc.desc      Testing for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList01', Level.LEVEL1, async (done:Function) => {
      console.info('[testList01] START');
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList01] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.listDirection).assertEqual('Axis.Vertical');
      expect(JSON.parse(strJson).$attrs.editMode).assertFalse();
      expect(JSON.parse(strJson).$attrs.edgeEffect).assertEqual('EdgeEffect.None');
      expect(JSON.parse(strJson).$attrs.chainAnimation).assertFalse();
      expect(JSON.parse(strJson).$attrs.divider.strokeWidth).assertEqual("2.00vp");
      expect(JSON.parse(strJson).$attrs.divider.color).assertEqual("#FFFFFFFF");
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("20.00vp");
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("20.00vp");
      console.info('[testList01] END');
      done()
    });

    /**
     * @tc.number    testList02
     * @tc.name      testList02
     * @tc.desc      Testing listDirection for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList02', Level.LEVEL1, async (done:Function) => {
      console.info('[testList02] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "listDirection": Axis.Horizontal
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList02] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList02] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList02] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.listDirection).assertEqual('Axis.Horizontal');
      console.info('testList02 END');
      done()
    });

    /**
     * @tc.number    testList03
     * @tc.name      testList03
     * @tc.desc      Testing editMode for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList03', Level.LEVEL1, async (done:Function) => {
      console.info('[testList03] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "editMode": true
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList03] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList03] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList03] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.editMode).assertTrue();
      console.info('testList03 END');
      done()
    });

    /**
     * @tc.number    testList04
     * @tc.name      testList04
     * @tc.desc      Testing edgeEffect for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList04', Level.LEVEL1, async (done:Function) => {
      console.info('[testList04] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "edgeEffect": EdgeEffect.Spring
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList04] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList04] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList04] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.edgeEffect).assertEqual('EdgeEffect.Spring');
      console.info('testList04 END');
      done()
    });

    /**
     * @tc.number    testList05
     * @tc.name      testList05
     * @tc.desc      Testing chainAnimation for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList05', Level.LEVEL1, async (done:Function) => {
      console.info('[testList05] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "chainAnimation": true
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList05] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList05] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList05] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.chainAnimation).assertTrue();
      console.info('testList05 END');
      done()
    });

    /**
     * @tc.number    testList06
     * @tc.name      testList06
     * @tc.desc      Testing strokeWidth for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList06', Level.LEVEL1, async (done:Function) => {
      console.info('[testList06] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "strokeWidth": "3.000000vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList06] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList06] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList06] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.strokeWidth).assertEqual("3.00vp");
      console.info('testList06 END');
      done()
    });

    /**
     * @tc.number    testList07
     * @tc.name      testList07
     * @tc.desc      Testing color for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList07', Level.LEVEL1, async (done:Function) => {
      console.info('[testList07] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "color": "#FF0000FF"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList07] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList07] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList07] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.color).assertEqual("#FF0000FF");
      console.info('testList07 END');
      done()
    });

    /**
     * @tc.number    testList08
     * @tc.name      testList08
     * @tc.desc      Testing startMargin for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList08', Level.LEVEL1, async (done:Function) => {
      console.info('[testList08] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startMargin": "30.000000vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList08] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList08] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList08] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("30.00vp");
      console.info('testList08 END');
      done()
    });

    /**
     * @tc.number    testList09
     * @tc.name      testList09
     * @tc.desc      Testing endMargin for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList09', Level.LEVEL1, async (done:Function) => {
      console.info('[testList09] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endMargin": "30.000000vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList09] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList09] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList09] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("30.00vp");
      console.info('testList09 END');
      done()
    });

    /**
     * @tc.number    testList10
     * @tc.name      testList10
     * @tc.desc      Testing editMode for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList10', Level.LEVEL1, async (done:Function) => {
      console.info('[testList10] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "editMode": "true123"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 901,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList10] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList10] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList10] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.editMode).assertTrue();
      console.info('testList10 END');
      done()
    });

    /**
     * @tc.number    testList11
     * @tc.name      testList11
     * @tc.desc      Testing editMode for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList11', Level.LEVEL1, async (done:Function) => {
      console.info('[testList11] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "editMode": "39000000"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 902,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList11] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList11] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList11] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.editMode).assertTrue();
      console.info('testList11 END');
      done()
    });

    /**
     * @tc.number    testList12
     * @tc.name      testList12
     * @tc.desc      Testing chainAnimation for the List component.
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('testList12', Level.LEVEL1, async (done:Function) => {
      console.info('[testList12] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "chainAnimation": "true123"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 903,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[testList12] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[testList12] change component data error: " + err.message);
      }
      await CommonFunc.sleep(4000);
      let strJson: string = getInspectorByKey('list_edit');

      console.info("[testList12] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.chainAnimation).assertTrue();
      console.info('testList12 END');
      done()
    });
  })
}
