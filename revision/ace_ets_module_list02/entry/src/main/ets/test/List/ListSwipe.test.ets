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
import { Driver, Component, ON } from '@ohos.UiTest'
import events_emitter from '@ohos.events.emitter';

export default function ListSwipeTest() {
  const SUITE: string = 'ListSwipe'
  const waitUiReadyMs = 1500

  describe('ListSwipeTest', () => {
    beforeEach(async (done: Function) => {
      let options: router.RouterOptions = {
        url: "MainAbility/pages/List/ListSwipe",
      }
      try {
        router.clear();
        let pages: router.RouterState = router.getState();
        if (pages == null || !("ListSwipe" == pages.name)) {
          await router.pushUrl(options).then(() => {
            console.info(`${SUITE} router.pushUrl success`);
          })
        }
      } catch (err) {
        console.error(`${SUITE} beforeEach error:` + JSON.stringify(err));
      }
      done()
    });

    /*
     * @tc.number :ArkUI_List_Swipe_0100
     * @tc.name   :test onStateChange callback when swipe left
     * @tc.desc   :swipe left to EXPANDED state, end onStateChange will be called
     * @tc.level  :Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Swipe_0100', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Swipe_0100'
      console.info(`${SUITE} ${CASE} START`);
      let driver: Driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)
      // 3.swipe item to left
      let baseKey: string = 'MyList1_ListItem0'
      let strJson: string = getInspectorByKey(baseKey);
      console.info("windowRect Left: " + globalThis.winLeft + ',' + "windowRect Top: " + globalThis.winTop);
      let rectCenter: number[] = getRectCenter(JSON.parse(strJson).$rect)
      await driver.swipe(Math.round(globalThis.winLeft + rectCenter[0] + (rectCenter[3] - rectCenter[2]) / 4), Math.round( globalThis.winTop + rectCenter[1]),
        0, Math.round(globalThis.winTop + rectCenter[1]))

      await sleep(8000) // wait for ui update
      console.info('get getInspectorByKey swipe0100')
      let strJson2: string = getInspectorByKey('swipe0100')
      console.info('ArkUI_List_Swipe_0100 text: ' + JSON.stringify(JSON.parse(strJson2)))
      expect(JSON.parse(strJson2).$type).assertEqual('Text')
      console.info("obj2.$attrs.content: " + JSON.parse(strJson2).$attrs.content)
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('end EXPANDED')
      console.info(`${SUITE} ${CASE} END`);
      done()
    });

    /*
     * @tc.number :ArkUI_List_Swipe_0300
     * @tc.name   :test onStateChange callback when swipe right
     * @tc.desc   :swipe right to EXPANDED state, end onStateChange will be called
     * @tc.level  :Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Swipe_0300', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Swipe_0300'
      console.info(`${SUITE} ${CASE} START`);
      let driver: Driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)

      let baseKey: string = 'MyList2_ListItem0'
      let strJson: string = getInspectorByKey(baseKey);
      console.info("windowRect Left: " + globalThis.winLeft + ',' + "windowRect Top: " + globalThis.winTop);
      // 2.swipe right
      let rectCenter: number[] = getRectCenter(JSON.parse(strJson).$rect)
      await driver.swipe(Math.round(globalThis.winLeft + rectCenter[0]),
        Math.round(globalThis.winTop + rectCenter[1]),
        Math.round(globalThis.winLeft + rectCenter[0] + (rectCenter[3] -rectCenter[2]) / 4),
        Math.round(globalThis.winTop + rectCenter[1]))

      await sleep(8000) // wait for ui update
      console.info('get getInspectorByKey swipe0300')
      let strJson2: string = getInspectorByKey('swipe0300')
      console.info('ArkUI_List_Swipe_0300 text: ' + JSON.stringify(JSON.parse(strJson2)))
      expect(JSON.parse(strJson2).$type).assertEqual('Text')
      console.info("obj2.$attrs.content: " + JSON.parse(strJson2).$attrs.content)
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('EXPANDED')
      console.info(`${SUITE} ${CASE} END`);
      done()
    });

    /*
     * @tc.number :ArkUI_List_Swipe_0400
     * @tc.name   :test closeAllSwipeActions interface
     * @tc.desc   :collapse expand item in ArkUI_List_Swipe_0200 by closeAllSwipeActions, onFinish will be called
     * @tc.level  :Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Swipe_0400', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Swipe_0400'
      console.info(`${SUITE} ${CASE} START`);
      let driver: Driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)

      // 2.trigger button click, expect state change COLLAPSED and onFinish be called
      sendEventByKey('MyList2CloseButton', 10, null)
      await sleep(8000) // wait for ui update
      let strJson2: string = getInspectorByKey('swipe0400')
      console.info('ArkUI_List_Swipe_0400 text: ' + JSON.stringify(JSON.parse(strJson2)))
      expect(JSON.parse(strJson2).$type).assertEqual('Text')
      console.info("obj2.$attrs.content: " + JSON.parse(strJson2).$attrs.content)
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('onFinish')
      console.info(`${SUITE} ${CASE} END`);
      done()
    });

    /*
     * @tc.number :ArkUI_List_Swipe_0500
     * @tc.name   :test onOffsetChange when swipe left
     * @tc.desc   :swipe left to EXPANDED state, onOffsetChange will be less than 0.
     * @tc.level  :Level 1
     * @tc.type
     * @tc.size
     */
    it('ArkUI_List_Swipe_0500', Level.LEVEL1, async (done: Function) => {
      let CASE: string = 'ArkUI_List_Swipe_0500'
      console.info(`${SUITE} ${CASE} START`);
      let driver: Driver = Driver.create()
      await driver.delayMs(waitUiReadyMs)
      // 2.swipe item to left
      let baseKey: string = 'MyList1_ListItem1'
      let strJson: string = getInspectorByKey(baseKey);
      let rectCenter: number[] = getRectCenter(JSON.parse(strJson).$rect)
      console.info("windowRect Left: " + globalThis.winLeft + ',' + "windowRect Top: " + globalThis.winTop);
      await driver.swipe(Math.round(globalThis.winLeft + rectCenter[0] + (rectCenter[3] -rectCenter[2]) / 4), Math.round(globalThis.winTop + rectCenter[1]),
        0, Math.round(globalThis.winTop + rectCenter[1]))
      await sleep(8000) // wait for ui update
      let strJson2: string = getInspectorByKey('swipe0500')

      console.info('ArkUI_List_Swipe_0500 text: ' + JSON.stringify(JSON.parse(strJson2)))
      expect(JSON.parse(strJson2).$type).assertEqual('Text')
      console.info("obj2.$attrs.content: " + JSON.parse(strJson2).$attrs.content)
      expect(JSON.parse(strJson2).$attrs.content).assertEqual('swipe end')
      console.info(`${SUITE} ${CASE} END`);
      done()
    });
  })
}

async function sleep(time: number) {
  return new Promise((resolve: (value: string) => void, reject) => {
    setTimeout(() => {
      resolve("sleep")
    }, time)
  }).then(() => {
    console.info(`sleep ${time} over...`)
  })
}

function getRectCenter(rect: string) {
  let regexp: RegExp = new RegExp('\\[', 'g');
  let rtn = rect.replace(regexp, '').replace(regexp, '').split(',')
  return [
    Number.parseFloat(rtn[0].trim()) + (Number.parseFloat(rtn[2].trim()) - Number.parseFloat(rtn[0].trim())) / 2,
    Number.parseFloat(rtn[1].trim()) + (Number.parseFloat(rtn[3].trim()) - Number.parseFloat(rtn[1].trim())) / 2,
    Number.parseFloat(rtn[0].trim()),
    Number.parseFloat(rtn[2].trim()),
  ]
}

function getRectCenterVp(rect: string) {
  let regexp: RegExp = new RegExp('\\[', 'g');
  let rtn: string[] = rect.replace(regexp, '').replace(regexp, '').split(',')
  return [
    px2vp(Number.parseFloat(rtn[0].trim()) + (Number.parseFloat(rtn[2].trim()) - Number.parseFloat(rtn[0].trim())) / 2),
    px2vp(Number.parseFloat(rtn[1].trim()) + (Number.parseFloat(rtn[3].trim()) - Number.parseFloat(rtn[1].trim())) / 2),
  ]
}