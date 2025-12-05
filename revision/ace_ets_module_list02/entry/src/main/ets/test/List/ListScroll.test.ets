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
import events_emitter from '@ohos.events.emitter';
import { BusinessError } from '@ohos.base'
import Utils from '../../MainAbility/common/MyUtils';

export default function ListScrollTest() {
  const SUITE: string = 'ListScroll'
  const waitUiReadyMs = 500;
  
  describe('ListScrollTest', () => {
    beforeEach(async (done: Function) => {
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/ListScroll",
      }
      try {
        router.clear();
        let pages: router.RouterState = router.getState();
        if (pages == null || !("ListScroll" == pages.name)) {
          await router.pushUrl(options).then(() => {
            console.info(`${SUITE} router.pushUrl success`);
          }).catch((err: BusinessError) => {
            console.error(`${SUITE} router.pushUrl failed, code is ${err.code}, message is ${err.message}`);
          })
          await Utils.sleep(2500);
        }
      } catch (err) {
        console.error(`${SUITE} beforeEach error:` + JSON.stringify(err));
      }
      done()
    });

    /*
     * @tc.number ArkUI_List_Scroll_0100
     * @tc.name   ArkUI_List_Scroll_0100
     * @tc.desc   Scroller.scrollToIndex(List) support animation
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Scroll_0100', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Scroll_0100'
      console.info(`${SUITE} ${CASE} START`)
      let callback = (event: events_emitter.EventData) => {
        expect(event.data?.start).assertEqual(14);
        done();
      }
      let event: events_emitter.InnerEvent = {
        eventId: 60000,
        priority: events_emitter.EventPriority.LOW
      }
      events_emitter.on(event, callback);
      let driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)
      sendEventByKey('ListScroll_Button_To_14', 10, "")
      console.info(`${SUITE} ${CASE} END`)
    });

    /*
     * @tc.number ArkUI_List_Scroll_0200
     * @tc.name   ArkUI_List_Scroll_0200
     * @tc.desc   Scroller.scrollToIndex(List) support animation
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Scroll_0200', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Scroll_0200'
      console.info(`${SUITE} ${CASE} START`)
      let callback = (event: events_emitter.EventData) => {
        expect(event.data?.start).assertEqual(15);
        done();
      }
      let event: events_emitter.InnerEvent = {
        eventId: 60001,
        priority: events_emitter.EventPriority.LOW
      }
      events_emitter.on(event, callback);
      let driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)
      sendEventByKey('ListScroll_Button_To_15', 10, "")
      console.info(`${SUITE} ${CASE} END`)
    });

    /*
     * @tc.number ArkUI_List_Scroll_0300
     * @tc.name   ArkUI_List_Scroll_0300
     * @tc.desc   Scroller.scrollToIndex(List) support animation
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Scroll_0300', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Scroll_0300'
      console.info(`${SUITE} ${CASE} START`)
      let callback = (event: events_emitter.EventData) => {
        expect(event.data?.start).assertEqual(0);
        done();
      }
      let event: events_emitter.InnerEvent = {
        eventId: 60002,
        priority: events_emitter.EventPriority.LOW
      }
      events_emitter.on(event, callback);
      let driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)
      sendEventByKey('ListScroll_Button_To_0', 10, "")
      console.info(`${SUITE} ${CASE} END`)
    });

    /*
     * @tc.number ArkUI_List_Scroll_0400
     * @tc.name   ArkUI_List_Scroll_0400
     * @tc.desc   Scroller.scrollToIndex(List) support animation
     * @tc.level  Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Scroll_0400', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Scroll_0400'
      console.info(`${SUITE} ${CASE} START`)
      let callback = (event: events_emitter.EventData) => {
        expect(event.data?.end).assertEqual(59);
        done();
      }
      let event: events_emitter.InnerEvent = {
        eventId: 60003,
        priority: events_emitter.EventPriority.LOW
      }
      events_emitter.on(event, callback);
      let driver: Driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)
      sendEventByKey('ListScroll_Button_To_max', 10, "")
      console.info(`${SUITE} ${CASE} END`)
    });
  })
}
