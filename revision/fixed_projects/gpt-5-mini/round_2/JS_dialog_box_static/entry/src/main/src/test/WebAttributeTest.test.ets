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

import { afterEach, beforeAll, beforeEach, describe, it, Level, Size, TestType } from '../../../hypium/index';
import Utils from './Util.test';
import { sendEventByKey } from '@internal.full.global';
import UIAbility from '@ohos.app.ability.UIAbility';
import { BusinessError } from '@ohos.base';
import common from '@ohos.app.ability.common';
import inspector from '@ohos.arkui.inspector';
import abilityDelegatorRegistry from '@ohos.app.ability.abilityDelegatorRegistry';
import { Driver, ON } from '@ohos.UiTest';

let testAbilityContext: common.UIAbilityContext;

export default function webAttributeTest() {

  describe('WebAttributeTest', () => {

    beforeAll(async (): Promise<void> => {
      await Utils.msSleep(2000)
      console.log("App start")
    });

    /*
    * @tc.number      : SUB_WEB_ONALERT_0100
    * @tc.name        : testOnAlertStatic001
    * @tc.desc        : test onAlert callback
    * @tc.size        : MediumTest
    * @tc.type        : Function
    * @tc.level       : Level 3
    */
    it('testOnAlertStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        await Utils.msSleep(2000);
        Utils.emitEvent("emitOnAlert001", 10)
        await Utils.msSleep(2000);
        Utils.registerEvent("emitOnAlert001", "alert test", 70017, done);
        inspector.sendEventByKey('WebAttributeTestButton', 10, '');
        done();
      })

    /*
    * @tc.number      : SUB_WEB_ONCONFIRM_0100
    * @tc.name        : testOnConfirmStatic001
    * @tc.desc        : test onConfirm normal
    * @tc.size        : MediumTest
    * @tc.type        : Function
    * @tc.level       : Level 3
    */
    it('testOnConfirmStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        await Utils.msSleep(2000);
        Utils.emitEvent("emitOnConfirm001", 10)
        await Utils.msSleep(2000);
        Utils.registerEvent("emitOnConfirm001", "confirm test", 70018, done);
        inspector.sendEventByKey('WebAttributeTestButton', 10, '');
        done();
      })

    /*
    * @tc.number      : SUB_WEB_ONPROMPT_0100
    * @tc.name        : testOnPromptStatic001
    * @tc.desc        : test onPrompt normal
    * @tc.size        : MediumTest
    * @tc.type        : Function
    * @tc.level       : Level 3
    */
    it('testOnPromptStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        await Utils.msSleep(2000);
        Utils.emitEvent("emitOnPrompt001", 10)
        await Utils.msSleep(2000);
        Utils.registerEvent("emitOnPrompt001", "prompt test", 70019, done);
        inspector.sendEventByKey('WebAttributeTestButton', 10, '');
        await Utils.msSleep(2000);
        done();
      })

    it('SwitchNextPageStatic', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
        await Utils.msSleep(1000);
        Utils.emitEvent('SwitchNextPageStatic', 10)
        await Utils.msSleep(1000);
        let driver: Driver = Driver.create();
        await Utils.msSleep(1000);
        let button = await driver.findComponent(ON.id('WebAttributeTestButton'));
        await Utils.msSleep(1000);
        button?.click();
        await Utils.msSleep(4000)
        done()
      })
  })
}
