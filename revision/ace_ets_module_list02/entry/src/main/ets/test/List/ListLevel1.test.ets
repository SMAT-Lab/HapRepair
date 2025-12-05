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
import {UiComponent, UiDriver, Component, Driver, UiWindow, ON, BY, WindowMode, PointerMatrix} from '@ohos.UiTest';

export default function listLevel1Test() {
  describe('listLevel1Test', () => {
    beforeEach(async (done: Function) => {
      console.info("listLevel1 beforeEach start");
      try {
        router.clear();
        let pages = router.getState();
        console.info("get listLevel1 state pages:" + JSON.stringify(pages));
        if (!("listTest" == pages.name)) {
          console.info("get listLevel1 pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl({url:"MainAbility/pages/List/ListLevel1"})
          await CommonFunc.sleep(2000);
          console.info("push listLevel1  page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push listLevel1 page error:" + err);
      }
      console.info("listLevel1 beforeEach end");
      done();
    });
    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("listLevel1 after each called");
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0100
     * @tc.desc      test list property
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0100', Level.LEVEL1, async (done: Function) => {
      console.info('[listLevel1Test_0100] START');
      await CommonFunc.sleep(2000);
      let json = getInspectorByKey('listlevel1');
      let Obj1: ESObject = JSON.parse(json);
      console.info("listlevel1 json info " + JSON.stringify(json));
      expect(Obj1.$attrs.divider.strokeWidth).assertEqual('2.00vp');
      expect(Obj1.$attrs.divider.color).assertEqual('#FFFFFFFF');
      expect(Obj1.$attrs.divider.startMargin).assertEqual('10.00vp');
      expect(Obj1.$attrs.divider.endMargin).assertEqual('10.00vp');
      expect(Obj1.$attrs.contentStartOffset).assertEqual('10.000000');
      expect(Obj1.$attrs.contentEndOffset).assertEqual('10.000000');
      console.info('[listLevel1Test_0100] END');
      done();
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0200
     * @tc.desc      test divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0200', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0200] START');
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
        console.info("[listLevel1Test_0200] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0200] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0200] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("30.00vp");
      console.info('listLevel1Test_0200 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0300
     * @tc.desc      test divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0300', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0300] START');
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
        console.info("[listLevel1Test_0300] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0300] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0300] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual('30.90vp');
      console.info('listLevel1Test_0300 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0400
     * @tc.desc      test divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0400', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0400] START');
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
        console.info("[listLevel1Test_0400] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0400] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0400] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("0.00vp");
      console.info('listLevel1Test_0400 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0500
     * @tc.desc      test divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0500', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0500] START');
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
        console.info("[listLevel1Test_0500] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0500] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0500] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("30.00vp");
      console.info('listLevel1Test_0500 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0600
     * @tc.desc      test divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0600', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0600] START');
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
        console.info("[listLevel1Test_0600] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0600] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0600] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("0.00vp");
      console.info('listLevel1Test_0600 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0700
     * @tc.desc      test divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0700', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0700] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endMargin": "30.90vp"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_0700] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0700] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0700] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("30.90vp");
      console.info('listLevel1Test_0700 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0800
     * @tc.desc      test divider startMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0800', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0800] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startMargin": "30%"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_0800] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0800] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0800] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.startMargin).assertEqual("0.00vp");
      console.info('listLevel1Test_0800 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_0900
     * @tc.desc      test divider endMargin
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_0900', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_0900] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endMargin": "5"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 125,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_0900] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_0900] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_0900] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.divider.endMargin).assertEqual("5.00vp");
      console.info('listLevel1Test_0900 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1000
     * @tc.desc      test contentStartOffset/contentEndOffset
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1000', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1000] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "contentStartOffset": "200",
            "contentEndOffset": "200"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1000] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1000] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_1000] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.contentStartOffset).assertEqual("200.000000");
      expect(JSON.parse(strJson).$attrs.contentEndOffset).assertEqual("200.000000");
      console.info('listLevel1Test_1000 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1100
     * @tc.desc      test contentEndOffset/padding
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1100', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1100] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "padding": "5",
            "contentEndOffset": "5"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1100] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1100] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_1100] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.padding).assertEqual("5.00vp");
      expect(JSON.parse(strJson).$attrs.contentEndOffset).assertEqual("5.000000");
      console.info('listLevel1Test_1100 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1200
     * @tc.desc      test contentStartOffset/contentEndOffset
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1200', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1200] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "padding": "0",
            "contentStartOffset": "5"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1200] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1200] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');

      console.info("[listLevel1Test_1200] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.padding).assertEqual("0.00vp");
      expect(JSON.parse(strJson).$attrs.contentStartOffset).assertEqual("5.000000");
      console.info('listLevel1Test_1200 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1300
     * @tc.desc      test contentStartOffset
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1300', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1300] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "padding": "0",
            "contentStartOffset": "5"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1300] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1300] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');
      console.info("[listLevel1Test_1300] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.padding).assertEqual("0.00vp");
      expect(JSON.parse(strJson).$attrs.contentStartOffset).assertEqual("5.000000");

      await CommonFunc.sleep(1000);
      let driver: Driver = Driver.create();
      let scrollColumn: Component = await driver.findComponent(ON.type('Scroll'));
      let button = await scrollColumn.scrollSearch(ON.id('GetItemRectButton'));
      await button.click();
      await CommonFunc.sleep(10000);
      let strJsonRectY: string = getInspectorByKey('itemRectY');
      let strJsonRectWidth: string = getInspectorByKey('itemRectWidth');
      let strJsonRectHeight: string = getInspectorByKey('itemRectHeight');
      console.info("[listLevel1Test_1300] JSON.parse(strJsonRectY) is: " + JSON.stringify(JSON.parse(strJsonRectY).$attrs.content));
      console.info("[listLevel1Test_1300] JSON.parse(strJsonRectWidth) is: " + JSON.stringify(JSON.parse(strJsonRectWidth).$attrs.content));
      console.info("[listLevel1Test_1300] JSON.parse(strJsonRectHeight) is: " + JSON.stringify(JSON.parse(strJsonRectHeight).$attrs.content));
      expect(JSON.parse(strJsonRectY).$attrs.content).assertEqual('itemRectY:5');
      expect(JSON.parse(strJsonRectWidth).$attrs.content).assertEqual('itemRectWidth:200');
      expect(JSON.parse(strJsonRectHeight).$attrs.content).assertEqual('itemRectHeight:100');
      console.info('listLevel1Test_1300 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1400
     * @tc.desc      test contentStartOffset/contentEndOffset and currentOffset
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1400', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1400] START');
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "contentStartOffset": "10",
            "contentEndOffset": "10"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 124,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1400] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1400] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listlevel1');
      console.info("[listLevel1Test_1400] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.contentStartOffset).assertEqual("10.000000");
      expect(JSON.parse(strJson).$attrs.contentEndOffset).assertEqual("10.000000");

      let driver: Driver = Driver.create();
      let scrollColumn: Component = await driver.findComponent(ON.type('Scroll'));
      let button = await scrollColumn.scrollSearch(ON.id('currentOffsetButton'));
      await button.click();
      await CommonFunc.sleep(1000);
      let strJsonX: string = getInspectorByKey('xoffset');
      let strJsonY: string = getInspectorByKey('yoffset');
      await CommonFunc.sleep(1000);
      console.info("[listLevel1Test_1400] component xoffset is: " + JSON.stringify(JSON.parse(strJsonX).$attrs.content));
      console.info("[listLevel1Test_1400] component yoffset is: " + JSON.stringify(JSON.parse(strJsonY).$attrs.content));
      expect(JSON.parse(strJsonX).$attrs.content).assertEqual("xoffset:0");
      expect(JSON.parse(strJsonY).$attrs.content).assertEqual("yoffset:-10");
      console.info('listLevel1Test_1400 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1500
     * @tc.desc      test ListItem swipeAction
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1500', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1500] START');
      let listItemRect = CommonFunc.getComponentRectOnScreen('listitemlevel0');
      let top = listItemRect.top;
      let right = listItemRect.right;
      let left = listItemRect.left;
      let bottom = listItemRect.bottom;
      console.info("listLevel1Test_1500 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom - top;
      let width = right - left;
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(right - vp2px(10)),Math.round(bottom - height/2),Math.round(left + width/2),Math.round(bottom - height/2));
      await CommonFunc.sleep(3000);
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startDeleteAreaDistance": "50"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 126,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1500] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1500] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listitemlevel1');
      console.info("[listLevel1Test_1500] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.startDeleteAreaDistance).assertEqual("50.00vp");
      console.info('listLevel1Test_1500 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1600
     * @tc.desc      test ListItem swipeAction
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1600', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1600] START');
      let listItemRect = CommonFunc.getComponentRectOnScreen('listitemlevel0');
      let top = listItemRect.top;
      let right = listItemRect.right;
      let left = listItemRect.left;
      let bottom = listItemRect.bottom;
      console.info("listLevel1Test_1600 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom - top;
      let width = right - left;
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(left + vp2px(10)),Math.round(bottom - height/2),Math.round(left + width/2),Math.round(bottom - height/2));
      await CommonFunc.sleep(3000);
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "endDeleteAreaDistance": "50"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 126,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1600] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1600] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listitemlevel1');
      console.info("[listLevel1Test_1600] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.endDeleteAreaDistance).assertEqual("50.00vp");
      console.info('listLevel1Test_1600 END');
      done()
    });

    /**
     * @tc.number    SUB_ACE_listLevel1Test_1700
     * @tc.desc      test ListItem swipeAction
     * @tc.level     Level 1
     * @tc.type
     * @tc.size
     */
    it('listLevel1Test_1700', Level.LEVEL1, async (done:Function) => {
      console.info('[listLevel1Test_1700] START');
      let listItemRect = CommonFunc.getComponentRectOnScreen('listitemlevel0');
      let top = listItemRect.top;
      let right = listItemRect.right;
      let left = listItemRect.left;
      let bottom = listItemRect.bottom;
      console.info("listLevel1Test_1700 value is" + left + " --" + right + "--" + top + "--" + bottom);
      let height = bottom - top;
      let width = right - left;
      let driver = await Driver.create()
      await CommonFunc.sleep(1000);
      await driver.swipe(Math.round(right - vp2px(10)),Math.round(bottom - height/2),Math.round(left + width/2),Math.round(bottom - height/2));
      await CommonFunc.sleep(3000);
      try {
        let eventData: events_emitter.EventData = {
          data: {
            "startDeleteAreaDistance": "100"
          }
        }
        let innerEvent: events_emitter.InnerEvent = {
          eventId: 126,
          priority: events_emitter.EventPriority.LOW
        }
        console.info("[listLevel1Test_1700] start to publish emit:" + JSON.stringify(eventData.data));
        events_emitter.emit(innerEvent, eventData);
      } catch (err) {
        console.log("[listLevel1Test_1700] change component data error: " + err.message);
      }
      await CommonFunc.sleep(1000);
      let strJson: string = getInspectorByKey('listitemlevel1');
      console.info("[listLevel1Test_1700] JSON.parse(strJson) is: " + JSON.stringify(JSON.parse(strJson)));
      expect(JSON.parse(strJson).$attrs.startDeleteAreaDistance).assertEqual("100.00vp");
      console.info('listLevel1Test_1700 END');
      done()
    });
  })
}
