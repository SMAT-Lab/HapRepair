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

import Utils from '../../MainAbility/common/Utils';
import CommonTest from '../common/CommonTest';
import { describe, expect } from '@ohos/hypium'

class PageConfig {
  testName: string = '';
  pageName: string = '';
  pageUrl: string = '';
  parentWidth: number = 0;
  parentHeight: number = 0;
  targetWidth: number = 0;
  targetHeight: number = 0;
}

class TestValue {
  describe: string = '';
  setValue: ESObject;
  expectValue: ESObject;
}

export default function flexBasisTest() {

  let supportView = [
    'Ellipse',
    'Line',
    'Polyline',
    'Polygon',
    'Path',
    'Rect',
    'Shape'
  ]

  let containerView = ['Row', 'Column', 'FlexRow', 'FlexRowReverse', 'FlexColumn', 'FlexColumnReverse']

  let parentWidth = 300;
  let parentHeight = 300;
  let targetWidth = 120;
  let targetHeight = 120;

  let pageConfig: PageConfig = {
    testName: 'testFlexBasis',
    pageName: 'FlexBasisPage',
    pageUrl: 'MainAbility/pages/flex/FlexBasisPage',
    parentWidth: parentWidth,
    parentHeight: parentHeight,
    targetWidth: targetWidth,
    targetHeight: targetHeight
  }

  let testValues: TestValue[] = [{
    describe: 'SetNumber',
    setValue: {
      flexBasis: 100.00
    },
    expectValue: {
      expectFlexBasis: '100.00vp',
      expectSize: vp2px(100.00)
    }
  }, {
    describe: 'SetStringPx',
    setValue: {
      flexBasis: '200.00px'
    },
    expectValue: {
      expectFlexBasis: '200.00px',
      expectSize: 200.00
    }
  }, {
    describe: 'SetStringPercent',
    setValue: {
      flexBasis: '50%'
    },
    expectValue: {
      expectSize: vp2px(120)
    }
  }]

  describe('FlexBasisTest', () => {
    CommonTest.initTestByContainerView(pageConfig, containerView, supportView, testValues, async (data: ESObject) => {

      //Verify the flexBasis is or not be set success
      console.info('[' + data.caseTag + '] check the flexBasis is or not be set success');
      if (data.testValue.describe == 'SetNumber' || data.testValue.describe == 'SetStringPx') {
        expect(data.viewObj.$attrs.flexBasis).assertEqual(data.testValue.expectValue.expectFlexBasis);
        console.info('[' + data.caseTag + '] data.viewObj.$attrs.flexBasis' + data.viewObj.$attrs.flexBasis);
      }

      //Get target component rect
      console.info('[' + data.caseTag + '] Get target component rect');
      let rect = Utils.getComponentRect(data.pageConfig.componentKey);

      console.info('[' + data.caseTag + '] check the size of the component in the spindle ' +
        'direction has been changed by flexBasis ');

      //Verify the size is or not be the expect size
      if (data.pageConfig.containerView == 'Row' || data.pageConfig.containerView == 'FlexRow'
        || data.pageConfig.containerView == 'FlexRowReverse') {

        let realWidth = rect.right - rect.left;
        console.info('[' + data.caseTag + '] realWidth : ' + realWidth);
        if (data.pageConfig.componentKey == 'TextInput') {
          expect(Math.floor(Math.abs(realWidth - (data.testValue.expectValue.expectSize))) <= 2).assertTrue();
        } else {
          expect(Math.floor(Math.abs(realWidth - (data.testValue.expectValue.expectSize))) <= 1).assertTrue();
        }
      } else if (data.pageConfig.containerView == 'Column' || data.pageConfig.containerView == 'FlexColumn'
        || data.pageConfig.containerView == 'FlexColumnReverse') {

        let realHeight = rect.bottom - rect.top;
        console.info('[' + data.caseTag + '] realHeight : ' + realHeight);
        expect(Math.floor(Math.abs(realHeight - (data.testValue.expectValue.expectSize))) <= 1).assertTrue();
      }
    })
  })
}