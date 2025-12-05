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

import textAreaInputTest from './TextTest/textAreaInputTest.test';
import preventDefault from './Hyperlink/preventDefault.test';
import searchButton from './Search/searchButton.test';
import minMaxFontSize from './TextTest/minMaxFontSize.test';
import marqueeOptionsTest1 from './TextTest/marqueeOptions1.test';
import marqueeOptionsTest2 from './TextTest/marqueeOptions2.test';
import TextInputStopBackPressTest from './TextInputTest/TextInputStopBackPress.test';
import TextAreaStopBackPressTest from './TextAreaTest/TextAreaStopBackPress.test';
import SearchStopBackPressTest from './Search/SearchStopBackPress.test';
import addOrDelText from './TextAreaTest/addOrDelText.test';
import TextOnChangeOptionsTest from './TextChangeTest/TextOnChangeOptionsTest.test';
import TextOnWillChangeOptionsTest from './TextChangeTest/TextOnWillChangeOptionsTest.test';
import bindSelectionMenuTest from './SelectionMenuTest/bindSelectionMenuTest.test';
import textAreaEllipsisModeTest from './EllipsisModeTest/textAreaEllipsisModeTest.test';
import textInputEllipsisModeTest from './EllipsisModeTest/textInputEllipsisModeTest.test';
import searchFontScaleTest from './FontScaleTest/searchFontScaleTest.test';
import textAreaFontScaleTest from './FontScaleTest/textAreaFontScaleTest.test';
import textInputFontScaleTest from './FontScaleTest/textInputFontScaleTest.test';
import searchHalfLeadingTest from './HalfLeadingTest/searchHalfLeadingTest.test';
import textAreaHalfLeadingTest from './HalfLeadingTest/textAreaHalfLeadingTest.test';
import textInputHalfLeadingTest from './HalfLeadingTest/textInputHalfLeadingTest.test';


export default function testsuite() {
  textAreaInputTest();
  preventDefault();
  searchButton();
  minMaxFontSize();
  marqueeOptionsTest1();
  marqueeOptionsTest2(); 
  TextInputStopBackPressTest();
  TextAreaStopBackPressTest();
  SearchStopBackPressTest();
  addOrDelText();
  TextOnChangeOptionsTest();
  TextOnWillChangeOptionsTest();
  bindSelectionMenuTest();
  textAreaEllipsisModeTest();
  textInputEllipsisModeTest();
  searchFontScaleTest();
  textAreaFontScaleTest();
  textInputFontScaleTest();
  searchHalfLeadingTest();
  textAreaHalfLeadingTest();
  textInputHalfLeadingTest();
}