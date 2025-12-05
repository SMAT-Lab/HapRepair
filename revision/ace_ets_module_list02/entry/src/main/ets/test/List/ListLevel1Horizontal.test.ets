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
import CommonFunc from '../../MainAbility/common/Common';
import events_emitter from '@ohos.events.emitter';
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function listLevel1HorizontalTest() {
  describe('listLevel1HorizontalTest', () => {
    beforeEach(async (done: Function) => {
      console.info("listLevel1HorizontalTest beforeEach start");
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listLevel1HorizontalTest state pages:" + JSON.stringify(pages));
        if (!("listTest" == pages.name)) {
          console.info("get listLevel1HorizontalTest pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl({url:"MainAbility/pages/List/ListLevel1Horizontal"})
          await CommonFunc.sleep(2000);
          console.info("push listLevel1HorizontalTest  page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listLevel1HorizontalTest page error:" + err);
      }
      console.info("listLevel1HorizontalTest beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listLevel1HorizontalTest after each called");
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0100
     * @tc.name      listLevel1HorizontalTest_0100
     * @tc.desc      test Horizontal divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0100', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0100] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endMargin": "30.00vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0100] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0100] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0100] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("30.00vp");
      console.info('listLevel1HorizontalTest_0100 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0200
     * @tc.name      listLevel1HorizontalTest_0200
     * @tc.desc      test Horizontal divider strokeWidth
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0200', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0200] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "strokeWidth": "50.5vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0200] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0200] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0200] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.strokeWidth).assertEqual("50.50vp");
      console.info('listLevel1HorizontalTest_0200 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0300
     * @tc.name      listLevel1HorizontalTest_0300
     * @tc.desc      test Horizontal divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0300', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0300] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startMargin": "30.00vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0300] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0300] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0300] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("30.00vp");
      console.info('listLevel1HorizontalTest_0300 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0400
     * @tc.name      listLevel1HorizontalTest_0400
     * @tc.desc      test Horizontal divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0400', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0400] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startMargin": "0.00vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0400] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0400] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0400] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("0.00vp");
      console.info('listLevel1HorizontalTest_0400 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0500
     * @tc.name      listLevel1HorizontalTest_0500
     * @tc.desc      test Horizontal divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0500', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0500] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startMargin": "30.90vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0500] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0500] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0500] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("30.90vp");
      console.info('listLevel1HorizontalTest_0500 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0600
     * @tc.name      listLevel1HorizontalTest_0600
     * @tc.desc      test Horizontal divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0600', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0600] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endMargin": "0.00vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0600] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0600] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0600] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("0.00vp");
      console.info('listLevel1HorizontalTest_0600 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1HorizontalTest_0700
     * @tc.name      listLevel1HorizontalTest_0700
     * @tc.desc      test Horizontal divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1HorizontalTest_0700', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1HorizontalTest_0700] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endMargin": "30.9vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1HorizontalTest_0700] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1HorizontalTest_0700] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1Horizontal');

      console.info("[listLevel1HorizontalTest_0700] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("30.90vp");
      console.info('listLevel1HorizontalTest_0700 END');
      done()
    });
  })
}
