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

import { Constants } from './constants';

class Sample {
    public sampleMethod(): Sample;
    public sampleMethod(input: number): Sample;
    public sampleMethod(input?: number): Sample {
        return new Sample();
    }

}

function test_new_array(): void {
    let arr11: number[] = new Array(Constants.ARRAY_SIZE);
    let index = 0;
    let a2 = arr11[2];
    let a = 1;
    let b = 2;
    let c = Constants.INFO[a + b];
    let s = Constants.INFO[a];
    let n = Constants.INFO[3];
    let y = Constants.INFO['b'];
}

function testArray(): void {
    const arr13: number[] = new Array();
    arr13[0] = 1;
    let a13 = arr13[0] + 1;

}

function arrayExpr(): void {
    new Array<number>();
    new Array<string>();
    new Array<Sample>();
    new Array<string | Sample>(2);
    new Array();
}

function arrayLiteral(): void {
    let arr1: number[] = [1, 1.1, 2];
    const arr2 = ['a', 'c'];
    const arr3 = [new Sample()];
    const arr4 = [1, 'hello'];
    const arr5 = [];
    const arr6 = [1, 'hello', new Sample()];
}