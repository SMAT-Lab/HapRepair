/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { beforeAll, beforeEach, describe, it, TestType, Size, Level } from '../../../hypium/index';
import router from '@ohos.router';
import Utils from './Util.test';
import { BusinessError } from '@ohos.base';
import UIAbility from '@ohos.app.ability.UIAbility';
import abilityDelegatorRegistry from '@ohos.app.ability.abilityDelegatorRegistry';
import { Driver, ON } from '@ohos.UiTest';
import common from '@ohos.app.ability.common';
import inspector from '@ohos.arkui.inspector';

let testAbilityContext: common.UIAbilityContext;

export default function WebViewEnumAndFontLibrary() {

  describe('WebViewEnumAndFontLibraryTest', () => {

    beforeAll(async (): Promise<void> => {
      await Utils.msSleep(2000)
      console.log("App start")
    });

    beforeEach(async () => {
      await Utils.msSleep(5000);
      console.info('beforeEach start');
    })

    /*
    * @tc.number     : SUB_WEB_FILESELECTORMODE_0100
    * @tc.name       : testFileSelectorModeStatic001
    * @tc.desc       : test FileSelectorModeFileOpenMode
    * @tc.size       : MediumTest
    * @tc.type       : Function
    * @tc.level      : Level 3
    */
    it('testFileSelectorModeStatic001', TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL3,
      async (done: () => void): Promise<void> => {
      await Utils.msSleep(2000);
      Utils.emitEvent('emitFileSelectorModeFileOpenMode', 10)
      await Utils.msSleep(2000);
      Utils.registerEvent('testFileSelectorMode001', 0, 861, done);
      inspector.sendEventByKey('buttonEnumAndFontLibrary', 10, '');
      await Utils.msSleep(2000);
      done();
    })
  })
}
