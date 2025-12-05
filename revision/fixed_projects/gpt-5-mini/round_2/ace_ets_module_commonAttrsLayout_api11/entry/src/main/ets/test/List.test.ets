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

import girdContainerJsunit from './GridContainerTest/GridContainerJsunit.test';
import gridSettingsJsunit from './GridContainerTest/GridSettingsJsunit.test';
import layoutConstraintsJsunit from './layoutConstraintsTest/LayoutConstraintsJsunit.test';
import comSizeAbilityTest_01 from './comSizeAbility/comSizeAbilityTest_01.test';
import comSizeAbilityTest_02 from './comSizeAbility/comSizeAbilityTest_02.test';
import comSizeAbilityTest_03 from './comSizeAbility/comSizeAbilityTest_03.test';
import comSizeAbilityTest_04 from './comSizeAbility/comSizeAbilityTest_04.test';
import comSizeAbilityTest_05 from './comSizeAbility/comSizeAbilityTest_05.test';
import comSizeAbilityTest_06 from './comSizeAbility/comSizeAbilityTest_06.test';
import comSizeAbilityTest_07 from './comSizeAbility/comSizeAbilityTest_07.test';
import comSizeAbilityTest_08 from './comSizeAbility/comSizeAbilityTest_08.test';
import comSizeAbilityTest_09 from './comSizeAbility/comSizeAbilityTest_09.test';
import comSizeAbilityTest_10 from './comSizeAbility/comSizeAbilityTest_10.test';
import comSizeAbilityTest_11 from './comSizeAbility/comSizeAbilityTest_11.test';
import comSizeAbilityTest_12 from './comSizeAbility/comSizeAbilityTest_12.test';
import comSizeAbilityTest_13 from './comSizeAbility/comSizeAbilityTest_13.test';
import comSizeAbilityTest_15 from './comSizeAbility/comSizeAbilityTest_15.test';
import comSizeAbilityTest_16 from './comSizeAbility/comSizeAbilityTest_16.test';
import comSizeAbilityTest_17 from './comSizeAbility/comSizeAbilityTest_17.test';
import comSizeAbilityTest_19 from './comSizeAbility/comSizeAbilityTest_19.test';
import comSizeAbilityTest_20 from './comSizeAbility/comSizeAbilityTest_20.test';
import comSizeAbilityTest_22 from './comSizeAbility/comSizeAbilityTest_22.test';
import comSizeAbilityTest_23 from './comSizeAbility/comSizeAbilityTest_23.test';
import comSizeAbilityTest_24 from './comSizeAbility/comSizeAbilityTest_24.test';
import comSizeAbilityTest_25 from './comSizeAbility/comSizeAbilityTest_25.test';
import comSizeAbilityTest_26 from './comSizeAbility/comSizeAbilityTest_26.test';
import comSizeAbilityTest_28 from './comSizeAbility/comSizeAbilityTest_28.test';
import getListItemRect1 from './getItemRect/list/getListItemRect1.test';
import getListItemRect2 from './getItemRect/list/getListItemRect2.test';
import getListItemRect3 from './getItemRect/list/getListItemRect3.test';
import getListItemRect4 from './getItemRect/list/getListItemRect4.test';
import getListItemRect5 from './getItemRect/list/getListItemRect5.test';
import getListItemRect6 from './getItemRect/list/getListItemRect6.test';
import getListItemGroupRect1 from './getItemRect/list/getListItemGroupRect1.test';
import getListItemGroupRect2 from './getItemRect/list/getListItemGroupRect2.test';
import getListItemGroupRect3 from './getItemRect/list/getListItemGroupRect3.test';
import getListItemGroupRect4 from './getItemRect/list/getListItemGroupRect4.test';
import getListItemGroupRect5 from './getItemRect/list/getListItemGroupRect5.test';
import getListItemGroupRect6 from './getItemRect/list/getListItemGroupRect6.test';
import getGridItemRect1 from './getItemRect/grid/getGridItemRect1.test';
import getGridItemRect2 from './getItemRect/grid/getGridItemRect2.test';
import getGridItemRect3 from './getItemRect/grid/getGridItemRect3.test';
import getGridItemRect4 from './getItemRect/grid/getGridItemRect4.test';
import getGridItemRect5 from './getItemRect/grid/getGridItemRect5.test';
import getGridItemRect6 from './getItemRect/grid/getGridItemRect6.test';
import getScrollItemRect1 from './getItemRect/scroll/getScrollItemRect1.test';
import getScrollItemRect2 from './getItemRect/scroll/getScrollItemRect2.test';
import getScrollItemRect3 from './getItemRect/scroll/getScrollItemRect3.test';
import getScrollItemRect4 from './getItemRect/scroll/getScrollItemRect4.test';
import getScrollItemRect5 from './getItemRect/scroll/getScrollItemRect5.test';
import MeasureJsunit from './MeasureSizeTest/MeasureJsunit.test';
import onMeasureSize1 from './MeasureSizeTest/onMeasureSize1.test';
import onMeasureSize2 from './MeasureSizeTest/onMeasureSize2.test';
import onMeasureSize3 from './MeasureSizeTest/onMeasureSize3.test';
import backgroundJsunit from './BackgroundTest/BackgroundJsunit.test';
import onPlaceChildren1 from './PlaceChildrenTest/onPlaceChildren1.test';
import CustomLayoutOnLayoutTest from './CustomComponentLayout/CustomLayoutOnLayout.test';
import CustomLayoutGeometryInfoTest from './CustomComponentLayout/CustomLayoutGeometryInfoTest.test';
import CustomLayoutLayoutChildTest from './CustomComponentLayout/CustomLayoutLayoutChildTest.test';

export default function testsuite() {
  girdContainerJsunit();
  gridSettingsJsunit();
  layoutConstraintsJsunit();
  comSizeAbilityTest_01();
  comSizeAbilityTest_02();
  comSizeAbilityTest_03();
  comSizeAbilityTest_04();
  comSizeAbilityTest_05();
  comSizeAbilityTest_06();
  comSizeAbilityTest_07();
  comSizeAbilityTest_08();
  comSizeAbilityTest_09();
  comSizeAbilityTest_10();
  comSizeAbilityTest_11();
  comSizeAbilityTest_12();
  comSizeAbilityTest_13();
  comSizeAbilityTest_15();
  comSizeAbilityTest_16();
  comSizeAbilityTest_17();
  comSizeAbilityTest_19();
  comSizeAbilityTest_20();
  comSizeAbilityTest_22();
  comSizeAbilityTest_23();
  comSizeAbilityTest_24();
  comSizeAbilityTest_25();
  comSizeAbilityTest_26();
  comSizeAbilityTest_28();
  getListItemRect1();
  getListItemRect2();
  getListItemRect3();
  getListItemRect4();
  getListItemRect5();
  getListItemRect6();
  getListItemGroupRect1();
  getListItemGroupRect2();
  getListItemGroupRect3();
  getListItemGroupRect4();
  getListItemGroupRect5();
  getListItemGroupRect6();
  getGridItemRect1();
  getGridItemRect2();
  getGridItemRect3();
  getGridItemRect4();
  getGridItemRect5();
  getGridItemRect6();
  getScrollItemRect1();
  getScrollItemRect2();
  getScrollItemRect3();
  getScrollItemRect4();
  getScrollItemRect5();
  onMeasureSize1();
  onMeasureSize2();
  onMeasureSize3();
  MeasureJsunit();
  backgroundJsunit();
  onPlaceChildren1();
  CustomLayoutOnLayoutTest();
  CustomLayoutGeometryInfoTest();
  CustomLayoutLayoutChildTest();
}