/*
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

import { AttrsManager } from '../../MainAbility/common/AttrsManager';
import Utils from '../../MainAbility/common/Utils';
import { expect, it, Level, Size, TestType } from '@ohos/hypium'

export default class CommonTest {
  /**
   * init test
   * @param pageConfig   : page config
   * @param supportView  : support view
   * @param testValues   : test values
   * @param functions    : [0]:executeTest, [last] checkTestResult
   */
  static initTest(pageConfig: ESObject, supportView: string[], testValues: ESObject, ...functions: ESObject) {
    supportView.forEach(targetView => {

      //Obtain the last index of cases that need to be executed.
      let lastIndex = 0;
      testValues.forEach((temp: ESObject, index: number) => {
        if (temp.exceptView == undefined || temp.exceptView.indexOf(targetView) < 0) {
          lastIndex = index;
        }
      });

      testValues.forEach((testValue: ESObject, index: number) => {

        //If there are components that need to be excluded, do not generate cases.
        if (testValue.exceptView == undefined || testValue.exceptView.indexOf(targetView) < 0) {
          let currentIndex = index;
          let sharePage = true;
          if (pageConfig.sharePage != undefined) {
            sharePage = pageConfig.sharePage;
          }

          //Create case name.
          let caseTag = pageConfig.testName + targetView + testValue.describe;

          //If there is a specified container view.
          let containerView = '';
          if (pageConfig.containerView != undefined) {
            containerView = pageConfig.containerView;
            caseTag = pageConfig.testName + containerView + targetView + testValue.describe;
          }

          /**
           * @tc.number SUB_ACE_ETS_COMPONENT_COMMON_ATTRS_0100
           * @tc.name caseTag, like:testPositionAlphabetIndexerSetNumber
           * @tc.desc Test whether the state of different components
           *            after modifying their properties meets expectations
           */
          it(caseTag, TestType.FUNCTION | Size.MEDIUMTEST | Level.LEVEL0, async (done: Function) => {
            console.info('[' + caseTag + '] START');
            //Push page.
            console.info('[' + caseTag + ']' + ' test start, push page.');
            pageConfig['targetView'] = targetView;
            pageConfig['componentKey'] = targetView;
            pageConfig['containerView'] = containerView;
            pageConfig['testValue'] = testValue;
            await Utils.pushPage(pageConfig);
            try {
              //If there is no call back functions, do nothing.
              if (functions.length == 0) done();
              if (functions[0] instanceof Array) {
                functions = functions[0];
                if (functions.length == 0) done();
              }

              //Execute test
              let data: ESObject = {
                pageConfig: pageConfig,
                caseTag: caseTag,
                testValue: testValue,
              }
              if (functions.length > 1) {
                if (functions[0] instanceof Function) {
                  let customParams: ESObject = await functions[0](data);
                  //Set custom params to data, can be used by check test result.
                  if (customParams) {
                    data['customParams'] = customParams;
                  }
                } else {
                  console.info('[' + caseTag + '] :functions[0] is not a call back function, do nothing.');
                }
              } else {
                //If there is only an assertion function, automatically modify the status value
                console.info('[' + caseTag + '] setValue : ' + JSON.stringify(data.testValue.setValue));
                AttrsManager.change(data.caseTag, data.testValue.setValue);
                await Utils.sleep(1000);
              }

              //Check test result
              data['viewObj'] = Utils.getComponentByKey(pageConfig['componentKey']);
              await functions[functions.length - 1](data);

            } catch (err) {
              let msg = 'CommonTest.initTest execute ' + caseTag + ' error:' + err
              console.error(msg);
              console.error(err.stack);
              if (err.name != 'AssertException') {
                await CommonTest.back(caseTag, sharePage, currentIndex, lastIndex);
                expect(msg).assertTrue();
              }
            }

            await CommonTest.back(caseTag, sharePage, currentIndex, lastIndex);

            console.info('[' + caseTag + '] END');
            done();
          });
        }
      })
    });
  }

  static async back(caseTag: string, sharePage: boolean, currentIndex: number, lastIndex: number) {
    if (sharePage) {
      if (currentIndex == lastIndex) {
        console.info('[' + caseTag + ']' + ' last case, close page.');
        await Utils.back();
      } else {
        console.info('[' + caseTag + ']' + ' share page.');
        await Utils.sleep(1000);
      }
    } else {
      console.info('[' + caseTag + ']' + ' close page.');
      await Utils.back();
    }
  }

  static initTestByContainerView(pageConfig: ESObject, containerView: string[], supportView: string[], testValues: ESObject[], ...functions: Function[]) {
    containerView.forEach(container => {
      pageConfig['containerView'] = container;
      CommonTest.initTest(pageConfig, supportView, testValues, ...functions);
    });
  }
}