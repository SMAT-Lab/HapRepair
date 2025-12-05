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

import CommonFunc from '../MainAbility/common/Common';
import { describe, beforeAll, beforeEach, afterEach, afterAll, it, expect, Level } from '@ohos/hypium';
import router from '@ohos.router';
import { Component, Driver, UiWindow, ON, MatchPattern, DisplayRotation, ResizeDirection, WindowMode, PointerMatrix} from '@ohos.UiTest';
import events_emitter from '@ohos.events.emitter'

export default function OnNewParamRouter() {
  describe('OnNewParamRouter', () => {
    beforeEach(async (done: Function) => {
      console.info("OnNewParamRouter beforeEach start");
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/router/OnNewParamRouter',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get OnNewParamRouter state pages:" + JSON.stringify(pages));
        if (!("OnNewParamRouter" == pages.name)) {
          console.info("get OnNewParamRouter pages.name:" + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(2000);
          console.info("push OnNewParamRouter page result:" + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push OnNewParamRouter page error:" + err);
      }
      console.info("OnNewParamRouter beforeEach end");
      done();
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("OnNewParamRouter after each called");
    });


    /**
     * @tc.number    OnNewParamRouter_0100
     * @tc.name      OnNewParamRouter_0100
     * @tc.desc      Test Router OnNewParam by pushUrl router.RouterMode.Single
     * @tc.level     Level 2
     * @tc.type
     * @tc.size
     */
    it('OnNewParamRouter_0100', Level.LEVEL2, async (done: Function) => {
      console.info('OnNewParamRouter_0100 START');
      let indexEvent: events_emitter.InnerEvent = {
        eventId: 51,
        priority: events_emitter.EventPriority.LOW
      }
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await driver.delayMs(500);

      let btn1: Component = await driver.findComponent(ON.id('pushId'));
      await btn1.click()
      console.info('[OnNewParamRouter_0100] btn1 click');
      await driver.delayMs(1000);

      let callback = (indexEvent: events_emitter.EventData) => {
        console.info('OnNewParamRouter_0100 get state result is: ' + JSON.stringify(indexEvent))
        try {
          expect(indexEvent.data?.paramString).assertEqual('{"msg":"pushUrl One Single"}');
        } catch (err) {
          console.info('OnNewParamRouter_0100 get state result is err: ' + JSON.stringify(err));
        }
        console.info('[OnNewParamRouter_0100] END');
        done();
      }
      try {
        events_emitter.on(indexEvent, callback);
      } catch (err) {
        console.info('OnNewParamRouter_0100 on events_emitter err : ' + JSON.stringify(err));
      }

      let btn2: Component = await driver.findComponent(ON.id('pushUrlId'));
      await btn2.click()
      console.info('[OnNewParamRouter_0100] btn2 click');
    });

    /**
     * @tc.number    OnNewParamRouter_0200
     * @tc.name      OnNewParamRouter_0200
     * @tc.desc      Test Router OnNewParam by replaceUrl router.RouterMode.Single
     * @tc.level     Level 2
     */
    it('OnNewParamRouter_0200', Level.LEVEL2, async (done: Function) => {
      console.info('OnNewParamRouter_0200 START');
      let indexEvent: events_emitter.InnerEvent = {
        eventId: 52,
        priority: events_emitter.EventPriority.LOW
      }
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await driver.delayMs(500);

      let btn1: Component = await driver.findComponent(ON.id('pushId'));
      await btn1.click()
      console.info('[OnNewParamRouter_0100] btn1 click');
      await driver.delayMs(1000);

      let callback = (indexEvent: events_emitter.EventData) => {
        console.info('OnNewParamRouter_0200 get state result is: ' + JSON.stringify(indexEvent))
        try {
          expect(indexEvent.data?.paramString).assertEqual('"test string"');
        } catch (err) {
          console.info('OnNewParamRouter_0200 get state result is err: ' + JSON.stringify(err));
        }
        console.info('[OnNewParamRouter_0200] END');
        done();
      }
      try {
        events_emitter.on(indexEvent, callback);
      } catch (err) {
        console.info('OnNewParamRouter_0200 on events_emitter err : ' + JSON.stringify(err));
      }

      let btn2: Component = await driver.findComponent(ON.id('replaceUrlId'));
      await btn2.click()
      console.info('[OnNewParamRouter_0200] btn2 click');
    });

    /**
     * @tc.number    OnNewParamRouter_0300
     * @tc.name      OnNewParamRouter_0300
     * @tc.desc      Test Router OnNewParam by pushNamedRoute router.RouterMode.Single
     * @tc.level     Level 2
     */
    it('OnNewParamRouter_0300', Level.LEVEL2, async (done: Function) => {
      console.info('OnNewParamRouter_0300 START');
      let indexEvent: events_emitter.InnerEvent = {
        eventId: 53,
        priority: events_emitter.EventPriority.LOW
      }
      await CommonFunc.sleep(500);

      let driver = Driver.create();
      await driver.delayMs(500);

      let btn1: Component = await driver.findComponent(ON.id('pushId'));
      await btn1.click()
      console.info('[OnNewParamRouter_0100] btn1 click');
      await driver.delayMs(1000);

      let callback = (indexEvent: events_emitter.EventData) => {
        console.info('OnNewParamRouter_0300 get state result is: ' + JSON.stringify(indexEvent))
        try {
          expect(indexEvent.data?.paramString).assertEqual('20');
        } catch (err) {
          console.info('OnNewParamRouter_0300 get state result is err: ' + JSON.stringify(err));
        }
        console.info('[OnNewParamRouter_0300] END');
        done();
      }
      try {
        events_emitter.on(indexEvent, callback);
      } catch (err) {
        console.info('OnNewParamRouter_0300 on events_emitter err : ' + JSON.stringify(err));
      }

      let btn2: Component = await driver.findComponent(ON.id('pushNamedId'));
      await btn2.click()
      console.info('[OnNewParamRouter_0300] btn2 click');
    });

  })
}