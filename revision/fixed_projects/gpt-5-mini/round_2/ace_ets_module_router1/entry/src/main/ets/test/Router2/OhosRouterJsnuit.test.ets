/**
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
import CommonFunc from '../../MainAbility/utils/Common';
import events_emitter from '@ohos.events.emitter';

export default function OhosRouterJsunit() {
  describe('ohosRouterJsunit', () => {
    beforeEach(async (done: Function) => {
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/ohosRouterA',
      }
      try {
        router.clear();
        let pages = router.getState();
        console.info("get ohosRouter state success " + JSON.stringify(pages));
        if (!("ohosRouterA" == pages.name)) {
          console.info("get ohosRouter state success " + JSON.stringify(pages.name));
          let result = await router.pushUrl(options);
          await CommonFunc.sleep(4000);
          console.info("push ohosRouter page success " + JSON.stringify(result));
        }
      } catch (err) {
        console.error("push ohosRouter page error " + err);
      }
      done()
    });

    afterEach(async () => {
      await CommonFunc.sleep(1000);
      console.info("ohosRouter after each called");
    });

    /**
     * @tc.number ohosRouterTest001
     * @tc.name ohosRouterTest001
     * @tc.desc
     */
    it('ohosRouterTest001', Level.LEVEL2, async (done: Function) => {
      console.info('[ohosRouterTest001] START');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/ohosRouterA',
      }
      try {
        let result = await router.pushUrl(options);
        await CommonFunc.sleep(1000);
        console.info("[ohosRouterTest001] push page success " + JSON.stringify(result));
        let pages = router.getState();
        expect(pages.index).assertEqual(3);
        expect(pages.name).assertEqual('ohosRouterA');
        expect(pages.path).assertEqual('MainAbility/pages/Router2/');
      } catch (err) {
        console.error("[ohosRouterTest001] push page error: " + err);
      }
      await CommonFunc.sleep(2000);
      done();
    });

    /**
     * @tc.number ohosRouterTest002
     * @tc.name ohosRouterTest002
     * @tc.desc
     */
    it('ohosRouterTest002', Level.LEVEL2, async (done: Function) => {
      console.info('[ohosRouterTest002] START');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/ohosRouterA'
      }
      try {
        let result = await router.pushUrl(options);
        console.info("[ohosRouterTest002] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[ohosRouterTest002] push page error " + JSON.stringify(err));
      }
      await CommonFunc.sleep(1500);
      let callback = (paramsEvent: events_emitter.EventData) => {
        console.info("[ohosRouterTest002] get state result is: " + JSON.stringify(paramsEvent));
        expect(paramsEvent.data?.params).assertEqual('B Page');
      }
      let paramsEvent: events_emitter.InnerEvent = {
        eventId: 142,
        priority: events_emitter.EventPriority.LOW
      }
      try {
        events_emitter.on(paramsEvent, callback);
      } catch (err) {
        console.info("[ohosRouterTest002] on events_emitter err : " + JSON.stringify(err));
      }
      console.info("[ohosRouterTest002] click push button result is: "
        + JSON.stringify(sendEventByKey('pushButtonA', 10, "")));
      await CommonFunc.sleep(1000);
      router.back();
      await CommonFunc.sleep(1000);
      done();
    });

    /**
     * @tc.number ohosRouterTest003
     * @tc.name ohosRouterTest003
     * @tc.desc
     */
    it('ohosRouterTest003', Level.LEVEL2, async (done: Function) => {
      console.info('[ohosRouterTest003] START');
      await CommonFunc.sleep(1000);
      router.clear();
      await CommonFunc.sleep(2000);
      let size = router.getLength();
      console.info("[ohosRouterTest003] size" + size);
      expect(size).assertEqual("1");
      await CommonFunc.sleep(1000);
      done();
    });

    /**
     * @tc.number ohosRouterTest004
     * @tc.name ohosRouterTest004
     * @tc.desc
     */
    it('ohosRouterTest004', Level.LEVEL2, async (done: Function) => {
      console.info('[ohosRouterTest004] START');
      router.clear();
      await CommonFunc.sleep(1000);
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/ohosRouterA'
      }
      try {
        let result = await router.pushUrl(options);
        console.info("[ohosRouterTest004] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[ohosRouterTest004] push page error " + JSON.stringify(err));
      }
      await CommonFunc.sleep(2000);
      let optionsB: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/ohosRouterB'
      }
      try {
        let result = await router.replace(optionsB);
        console.info("[ohosRouterTest004] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[ohosRouterTest004] push page error " + JSON.stringify(err));
      }
      await CommonFunc.sleep(1500);
      let size = router.getLength();
      console.info("[ohosRouterTest004] size" + size);
      expect(size).assertEqual("2");
      done();
    });

    /**
     * @tc.number ohosRouterTest005
     * @tc.name ohosRouterTest005
     * @tc.desc
     */
    it('ohosRouterTest005', Level.LEVEL2, async (done: Function) => {
      console.info('[ohosRouterTest005] START');
      let options: router.RouterOptions = {
        url: 'MainAbility/pages/Router2/ohosRouterA'
      }
      try {
        let result = await router.pushUrl(options);
        console.info("[ohosRouterTest005] push page success " + JSON.stringify(result));
      } catch (err) {
        console.error("[ohosRouterTest005] push page error " + JSON.stringify(err));
      }
      await CommonFunc.sleep(1000);
      router.clear();
      await CommonFunc.sleep(1500);
      let callback = (paramsEvent: events_emitter.EventData) => {
        console.info("[ohosRouterTest005] get state result is: " + JSON.stringify(paramsEvent));
        expect(paramsEvent.data?.params).assertEqual('B Page');
      }
      let paramsEvent: events_emitter.InnerEvent = {
        eventId: 142,
        priority: events_emitter.EventPriority.LOW
      }
      try {
        events_emitter.on(paramsEvent, callback);
      } catch (err) {
        console.info("[ohosRouterTest005] on events_emitter err : " + JSON.stringify(err));
      }
      console.info("[ohosRouterTest005] click replace button result is: "
        + JSON.stringify(sendEventByKey('ReplaceButtonA', 10, "")));
      done();
    });
  });
}