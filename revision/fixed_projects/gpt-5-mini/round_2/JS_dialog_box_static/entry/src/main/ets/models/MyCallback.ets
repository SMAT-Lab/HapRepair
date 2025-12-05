/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
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

import { AsyncCallback, BusinessError, Callback, ErrorCallback } from '@ohos.base';
import hilog from '@ohos.hilog';

const SYNTAX_ERROR_CODE = 1002;

export class MyCallback {
  static myCallback(callback: Callback<string>): void {
    hilog.info(0x0000, 'testTag', 'myCallback');
    callback('myCallback');
  }

  static myAsyncCallback(callback: AsyncCallback<string, string>): void {
    hilog.info(0x0000, 'testTag', 'myAsyncCallback');
    let error = new Error('Business Error', 'basicError2 message', undefined);
    let be2: BusinessError<string> = new BusinessError<string>(SYNTAX_ERROR_CODE, error);
    callback(be2, 'yAsyncCallback');
  }

  static myErrorCallback(callback: ErrorCallback<BusinessError<string>>): void {
    hilog.info(0x0000, 'testTag', 'myErrorCallback');
    let error = new Error('Business Error', 'myErrorCallback message', undefined);
    let result: string = 'test';
    let be: BusinessError<string> = new BusinessError<string>(SYNTAX_ERROR_CODE, result, error);
    callback(be);
  }

  static runCasesOfCallback() {
    MyCallback.myCallback((data: string) => {
      hilog.info(0x0000, 'testTag', 'MyCallback' + data);
    })
    MyCallback.myAsyncCallback((error: BusinessError<string>|null, result: string|undefined) => {
      hilog.info(0x0000, 'testTag', `myAsyncCallback, error: ${error?.message}, result:${result}`);
    })
    MyCallback.myErrorCallback((error: BusinessError<string>) => {
      hilog.info(0x0000, 'testTag', `myAsyncCallback, error: ${error.message}`);
    })
  }
}