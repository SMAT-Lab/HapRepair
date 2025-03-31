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

let globalValue: number = 0;

class BasicTest {
    public listeners: number[] = [];

    basicMethod(): void {
        const outer = 2;
        function nestedMethod(): void {
            let nested = 'abc';
        }
        nestedMethod();
    }

    basicOuterMethod1(): number {
        const output = 3;
        function basicNestedMethod1(input: number): number {
            return output + input;
        }
        return basicNestedMethod1(2);
    }

    basicOuterMethod2(index: number): void {
        this.listeners.forEach(listener => {
            console.log(index + listener);
        });
    }

    basicOuterMethod3(output: string): void {
        let basicNestedMethod3 = function(): void {
            console.log(output);
        };
        basicNestedMethod3();
    }
}

function basicOuterMethod4(): (input: number) => void {
    const base = 3;
    return function basicNestedMethod4(input: number): void {
        input += base;
    };
}

function callMethod4(): void {
    let callMethod = basicOuterMethod4();
    callMethod(3);
}

function outerFunction1(outerInput: number): void {
    let count = 0;
    let flag = 1;
    function innerFunction1(innerInput: string): string {
        count++;
        let result: string;
        switch (flag) {
            case 1:
                result = innerInput + 'ok1';
                break;
            case 2:
                result = innerInput + 'ok2';
                break;
            default:
                result = innerInput + 'no ok';
        }
        return result;
    }
    console.log(innerFunction1('abc'));

    let innerFunction2 = function(): void {
        console.log(outerInput);
    };
    innerFunction2();
}

class ClosureClass {
    public outerFunction2(outerInput: number): void {
        console.log(innerFunction2('abc'));
        function innerFunction2(outerInput: string): string {
            count = count + outerInput;
            for (let item of nums) {
                count = count + item;
            }
            return `${outerInput}: ${count}`;
        }
        let count = 'abc';
        let nums = [1, 2, 3, 4];
    }
}

namespace closureNamespace {
    function outerFunction3(outerInput: number): string {
        let count = 0;
        let size = 10;
        function innerFunction3(): string;
        function innerFunction3(innerInput: string): string;
        function innerFunction3(innerInput?: string): string {
            let res = count + size + globalValue;
            return `${outerInput}: ${res}`;
        }
        return innerFunction3();
    }

    class ClosureClass {
        public outerFunction3(outerInput: number): void {
            let flag = true;
            let res = 'no ok';
            innerFunction3();
            function innerFunction3(): void {
                if (!flag) {
                    return;
                }
                try {
                    while (outerInput > 0) {
                        outerInput--;
                    }
                } catch (error) {
                    console.log(error);
                }
            }
        }
    }
}