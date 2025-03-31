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

import {
    ANONYMOUS_METHOD_PREFIX, ArkStaticFieldRef,
    DEFAULT_ARK_CLASS_NAME,
    GlobalRef,
    LEXICAL_ENV_NAME_PREFIX,
    NAME_DELIMITER,
    NAME_PREFIX,
} from '../../../../src';

export const ArrowFunction_Expect_IR = {
    methods: [{
        name: '%dflt',
        stmts: [
            {
                text: 'this = this: @function/ArrowFunctionTest.ts: %dflt',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: 'func1 = %AM0',
                operandOriginalPositions: [
                    [16, 5, 16, 10], [16, 13, 16, 29],
                ],
            },
            {
                text: 'func2 = %AM1',
                operandOriginalPositions: [
                    [17, 5, 17, 10], [17, 13, 20, 2],
                ],
            },
            {
                text: 'func3 = %AM2',
                operandOriginalPositions: [
                    [21, 5, 21, 10], [21, 13, 21, 26],
                ],
            },
            {
                text: 'func4 = %AM3',
                operandOriginalPositions: [
                    [22, 5, 22, 10], [22, 13, 22, 31],
                ],
            },
            {
                text: 'return',
                operandOriginalPositions: [],
            },
        ],
    }, {
        name: '%AM0',
        stmts1: [
            'i = parameter0: number',
            'this = this: @function/ArrowFunctionTest.ts: %dflt',
            'return i',
        ],
        stmts: [
            {
                text: 'i = parameter0: number',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: 'this = this: @function/ArrowFunctionTest.ts: %dflt',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: 'return i',
                operandOriginalPositions: [
                    [16, 28, 16, 29],
                ],
            },
        ],
    }, {
        name: '%AM1',
        stmts1: [
            'this = this: @function/ArrowFunctionTest.ts: %dflt',
            'i = 0',
            'i = i + 1',
            'return',
        ],
        stmts: [
            {
                text: 'this = this: @function/ArrowFunctionTest.ts: %dflt',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: 'i = 0',
                operandOriginalPositions: [
                    [18, 9, 18, 10], [18, 13, 18, 14],
                ],
            },
            {
                text: 'i = i + 1',
                operandOriginalPositions: [
                    [19, 5, 19, 6], [19, 5, 19, 8], [19, 5, 19, 6], [-1, -1, -1, -1],
                ],
            },
            {
                text: 'return',
                operandOriginalPositions: [],
            },
        ],
    }, {
        name: '%AM2',
        stmts1: [
            'this = this: @function/ArrowFunctionTest.ts: %dflt',
            '%0 = staticinvoke <@%unk/%unk: .func2()>()',
            'return %0',
        ],
        stmts: [
            {
                text: 'this = this: @function/ArrowFunctionTest.ts: %dflt',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: '%0 = staticinvoke <@%unk/%unk: .func2()>()',
                operandOriginalPositions: [
                    [21, 19, 21, 26], [21, 19, 21, 26],
                ],
            },
            {
                text: 'return %0',
                operandOriginalPositions: [
                    [21, 19, 21, 26],
                ],
            },
        ],
    }, {
        name: '%AM3',
        stmts1: [
            'i = parameter0: number',
            'this = this: @function/ArrowFunctionTest.ts: %dflt',
            'i = i + 1',
            'return i',
        ],
        stmts: [
            {
                text: 'i = parameter0: number',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: 'this = this: @function/ArrowFunctionTest.ts: %dflt',
                operandOriginalPositions: [
                    null, null,
                ],
            },
            {
                text: 'i = i + 1',
                operandOriginalPositions: [
                    [22, 28, 22, 29], [22, 28, 22, 31], [22, 28, 22, 29], [-1, -1, -1, -1],
                ],
            },
            {
                text: 'return i',
                operandOriginalPositions: [
                    [22, 28, 22, 31],
                ],
            },
        ],
    }],
};

export const OverloadMethod_Expect_IR = {
    methodDeclareLines: [16, 17],
    methodDeclareSignatures: [
        {
            toString: '@function/OverloadFunctionTest.ts: %dflt.overloadedFunction1(number)',
            methodSubSignature: {
                returnType: 'string'
            }
        },
        {
            toString: '@function/OverloadFunctionTest.ts: %dflt.overloadedFunction1(string)',
            methodSubSignature: {
                returnType: 'number'
            }
        }

    ],
    line: 18,
    methodSignature: {
        toString: '@function/OverloadFunctionTest.ts: %dflt.overloadedFunction1(any)',
        methodSubSignature: {
            returnType: 'any'
        }
    },
    body: {
        locals: [
            {
                name: 'x'
            }
        ]
    }
};

export const OverloadClassMethod_Expect_IR = {
    methodDeclareLines: [29, 30, 31],
    methodDeclareSignatures: [
        {
            toString: '@function/OverloadFunctionTest.ts: OverloadClass.overloadedFunction2(number, number)',
            methodSubSignature: {
                returnType: 'string'
            }
        },
        {
            toString: '@function/OverloadFunctionTest.ts: OverloadClass.overloadedFunction2(string, string)',
            methodSubSignature: {
                returnType: 'number'
            }
        },
        {
            toString: '@function/OverloadFunctionTest.ts: OverloadClass.overloadedFunction2(string, string)',
            methodSubSignature: {
                returnType: 'string'
            }
        }
    ],
    line: 33,
    methodSignature: {
        toString: '@function/OverloadFunctionTest.ts: OverloadClass.overloadedFunction2(number|string, number|string)',
        methodSubSignature: {
            returnType: 'string|number'
        }
    },
    body: {
        locals: [
            {
                name: 'x'
            }
        ]
    }
};

export const OverloadNamespaceMethod_Expect_IR = {
    methodDeclareLines: [45, 46, 47],
    methodDeclareSignatures: [
        {
            toString: '@function/OverloadFunctionTest.ts: overloadNamespace.%dflt.overloadedFunction3(number)',
            methodSubSignature: {
                returnType: 'string'
            }
        },
        {
            toString: '@function/OverloadFunctionTest.ts: overloadNamespace.%dflt.overloadedFunction3(string)',
            methodSubSignature: {
                returnType: 'number'
            }
        },
        {
            toString: '@function/OverloadFunctionTest.ts: overloadNamespace.%dflt.overloadedFunction3(string)',
            methodSubSignature: {
                returnType: 'boolean'
            }
        }
    ],
    line: null,
    methodSignature: null
};

export const OverloadInterfaceMethod_Expect_IR = {
    methodDeclareLines: [51, 52],
    methodDeclareSignatures: [
        {
            toString: '@function/OverloadFunctionTest.ts: OverloadInterface.overloadedFunction4(number)',
            methodSubSignature: {
                returnType: 'number'
            }
        },
        {
            toString: '@function/OverloadFunctionTest.ts: OverloadInterface.overloadedFunction4(string)',
            methodSubSignature: {
                returnType: 'string'
            }
        }
    ],
    line: null,
    methodSignature: null,
};

export const NoOverloadMethod_Expect_IR = {
    methodDeclareLines: [55],
    methodDeclareSignatures: [
        {
            toString: '@function/OverloadFunctionTest.ts: %dflt.function5(string)',
            methodSubSignature: {
                returnType: 'number'
            }
        }
    ],
    line: null,
    methodSignature: null
};

export const NoOverloadMethodWithBody_Expect_IR = {
    methodDeclareLines: [57],
    methodDeclareSignatures: [
        {
            toString: '@function/OverloadFunctionTest.ts: %dflt.function6(number)',
            methodSubSignature: {
                returnType: 'number'
            }
        }
    ],
    line: 58,
    methodSignature: {
        toString: '@function/OverloadFunctionTest.ts: %dflt.function6(number)',
        methodSubSignature: {
            returnType: 'number'
        }
    },
    body: {
        locals: [
            {
                name: 'x'
            }
        ]
    }
};

export const NoOverloadMethodWithBody2_Expect_IR = {
    methodDeclareLines: null,
    methodDeclareSignatures: null,
    line: 62,
    methodSignature: {
        toString: '@function/OverloadFunctionTest.ts: %dflt.function7(number)',
        methodSubSignature: {
            returnType: 'number'
        }
    },
    body: {
        locals: [
            {
                name: 'x'
            }
        ]
    }
};

export const BasicNestedMethod_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: BasicTest.basicMethod()'
    },
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: BasicTest.%nestedMethod$basicMethod()`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: 'nested',
                type: 'string',
                declaringStmt: {
                    text: `nested = 'abc'`
                },
                usedStmts: []
            }
        ]
    }
};

export const BasicOuterMethod_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: BasicTest.basicMethod()`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: 'outer',
                type: 'number',
                declaringStmt: {
                    text: `outer = 2`
                },
                usedStmts: []
            },
        ],
        globals: [
            {
                name: 'nestedMethod',
                instanceof: GlobalRef,
                // TODO: 此处需要找到对应的ref
                ref: null,
                usedStmts: [
                    {
                        text: 'staticinvoke <@%unk/%unk: .nestedMethod()>()'
                    }
                ]
            }
        ],
        stmts: [
            {
                text: 'this = this: @function/ClosureParamsTest.ts: BasicTest',
            },
            {
                text: 'outer = 2',
            },
            {
                text: 'staticinvoke <@%unk/%unk: .nestedMethod()>()',
            },
            {
                text: 'return',
            }
        ]
    }
};

export const BasicNestedMethod1_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: BasicTest.basicOuterMethod1()'
    },
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: BasicTest.%basicNestedMethod1$basicOuterMethod1([output], number)`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `%closures0`,
                type: '[output]',
                declaringStmt: {
                    text: '%closures0 = parameter0: [output]'
                },
                usedStmts: [
                    {
                        text: 'output = %closures0.output'
                    }
                ]
            },
            {
                name: 'output',
                type: 'number',
                declaringStmt: {
                    text: `output = %closures0.output`
                },
                usedStmts: [
                    {
                        text: '%0 = output + input'
                    }
                ]
            },
            {
                name: `input`,
                type: 'number',
                declaringStmt: {
                    text: 'input = parameter1: number'
                },
                usedStmts: [
                    {
                        text: '%0 = output + input'
                    }
                ]
            },
        ]
    }
};

export const BasicOuterMethod1_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: BasicTest.basicOuterMethod1()`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: 'output',
                type: 'number',
                declaringStmt: {
                    text: `output = 3`
                },
                usedStmts: []
            },
            {
                name: '%closures0',
                type: '[output]',
                declaringStmt: null,
                usedStmts: [
                    {
                        text: '%0 = staticinvoke <@function/ClosureParamsTest.ts: BasicTest.%basicNestedMethod1$basicOuterMethod1([output], number)>(%closures0, 2)'
                    }
                ]
            }
        ],
        globals: [
            {
                name: 'basicNestedMethod1',
                instanceof: GlobalRef,
                ref: {
                    instanceof: ArkStaticFieldRef,
                    declaringSignature: '@function/ClosureParamsTest.ts: BasicTest',
                    fieldName: '%basicNestedMethod1$basicOuterMethod1',
                    type: 'closures: @function/ClosureParamsTest.ts: BasicTest.%basicNestedMethod1$basicOuterMethod1([output], number)',
                },
                usedStmts: [
                    {
                        text: '%0 = staticinvoke <@function/ClosureParamsTest.ts: BasicTest.%basicNestedMethod1$basicOuterMethod1([output], number)>(%closures0, 2)'
                    }
                ]
            }
        ],
        stmts: [
            {
                text: 'this = this: @function/ClosureParamsTest.ts: BasicTest',
            },
            {
                text: 'output = 3',
            },
            {
                text: '%0 = staticinvoke <@function/ClosureParamsTest.ts: BasicTest.%basicNestedMethod1$basicOuterMethod1([output], number)>(%closures0, 2)',
            },
            {
                text: 'return %0',
            }
        ]
    }
};

export const BasicNestedMethod2_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: BasicTest.basicOuterMethod2(number)'
    },
    methodSignature: {
        // TODO: 此处unknown类型可以从class的field类型获取，需要类型推导处理
        toString: `@function/ClosureParamsTest.ts: BasicTest.%AM0$basicOuterMethod2([index], unknown)`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: '%closures0',
                type: '[index]',
                declaringStmt: {
                    text: '%closures0 = parameter0: [index]'
                },
                usedStmts: [
                    {
                        text: 'index = %closures0.index'
                    }
                ]
            },
            {
                name: 'index',
                type: 'number',
                declaringStmt: {
                    text: `index = %closures0.index`
                },
                usedStmts: [
                    {
                        text: '%0 = index + listener'
                    }
                ]
            }
        ],
        globals: [
            {
                name: 'console',
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'instanceinvoke console.<@%unk/%unk: .log()>(%0)'
                    }
                ]
            }
        ]
    }
};

export const BasicOuterMethod2_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: BasicTest.basicOuterMethod2(number)`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `%closures0`,
                type: '[index]',
                declaringStmt: null,
                usedStmts: []
            },
            {
                name: 'index',
                type: 'number',
                declaringStmt: {
                    text: `index = parameter0: number`
                },
                usedStmts: []
            },
            {
                name: '%AM0$basicOuterMethod2',
                type: 'closures: @function/ClosureParamsTest.ts: BasicTest.%AM0$basicOuterMethod2([index], unknown)',
                declaringStmt: null,
                usedStmts: [
                    {
                        text: 'instanceinvoke %0.<@%unk/%unk: .forEach()>(%AM0$basicOuterMethod2)'
                    }
                ]
            }
        ],
        stmts: [
            {
                text: 'index = parameter0: number',
            },
            {
                text: 'this = this: @function/ClosureParamsTest.ts: BasicTest',
            },
            {
                text: '%0 = this.<@function/ClosureParamsTest.ts: BasicTest.listeners>',
            },
            {
                text: 'instanceinvoke %0.<@%unk/%unk: .forEach()>(%AM0$basicOuterMethod2)',
            },
            {
                text: 'return',
            }
        ]
    }
};

export const BasicNestedMethod3_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: BasicTest.basicOuterMethod3(string)',
    },
    methodSignature: {
        toString: '@function/ClosureParamsTest.ts: BasicTest.%AM1$basicOuterMethod3([output])',
        methodSubSignature: {
            parameters: [
                {
                    name: '%closures0',
                    type: '[output]'
                }
            ],
            returnType: 'void'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: '%closures0',
                type: '[output]',
                declaringStmt: {
                    text: '%closures0 = parameter0: [output]'
                },
                usedStmts: [
                    {
                        text: `output = %closures0.output`
                    }
                ]
            },
            {
                name: 'output',
                type: 'string',
                declaringStmt: {
                    text: 'output = %closures0.output'
                },
            },
            {
                name: 'this',
                type: '@function/ClosureParamsTest.ts: BasicTest',
            }
        ],
        globals: [
            {
                name: 'console',
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'instanceinvoke console.<@%unk/%unk: .log()>(output)'
                    }
                ]
            }
        ]
    }
};

export const BasicOuterMethod3_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: BasicTest.basicOuterMethod3(string)`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `%closures0`,
                type: '[output]',
                declaringStmt: null,
                usedStmts: [
                    {
                        text: 'ptrinvoke <@function/ClosureParamsTest.ts: BasicTest.%AM1$basicOuterMethod3([output])>(%closures0)'
                    }
                ]
            },
            {
                name: 'output',
                type: 'string',
                declaringStmt: {
                    text: `output = parameter0: string`
                },
                usedStmts: []
            },
            {
                name: '%AM1$basicOuterMethod3',
                type: 'closures: @function/ClosureParamsTest.ts: BasicTest.%AM1$basicOuterMethod3([output])',
                declaringStmt: null,
                usedStmts: [
                    {
                        text: 'basicNestedMethod3 = %AM1$basicOuterMethod3'
                    }
                ]
            },
            {
                name: 'basicNestedMethod3',
                type: 'closures: @function/ClosureParamsTest.ts: BasicTest.%AM1$basicOuterMethod3([output])',
                declaringStmt: {
                    text: 'basicNestedMethod3 = %AM1$basicOuterMethod3'
                },
                usedStmts: [
                    {
                        text: 'ptrinvoke <@function/ClosureParamsTest.ts: BasicTest.%AM1$basicOuterMethod3([output])>(%closures0)'
                    }
                ]
            }
        ],
        stmts: [
            {
                text: 'output = parameter0: string',
            },
            {
                text: 'this = this: @function/ClosureParamsTest.ts: BasicTest',
            },
            {
                text: 'basicNestedMethod3 = %AM1$basicOuterMethod3',
            },
            {
                text: 'ptrinvoke <@function/ClosureParamsTest.ts: BasicTest.%AM1$basicOuterMethod3([output])>(%closures0)',
            },
            {
                text: 'return',
            }
        ]
    }
};

export const BasicNestedMethod4_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: %dflt.basicOuterMethod4()',
    },
    methodSignature: {
        toString: '@function/ClosureParamsTest.ts: %dflt.%basicNestedMethod4$basicOuterMethod4([base], number)',
        methodSubSignature: {
            parameters: [
                {
                    name: '%closures0',
                    type: '[base]'
                },
                {
                    name: 'input',
                    type: 'number'
                }
            ],
            returnType: 'void'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: '%closures0',
                type: '[base]',
                declaringStmt: {
                    text: '%closures0 = parameter0: [base]'
                },
                usedStmts: [
                    {
                        text: `base = %closures0.base`
                    }
                ]
            },
            {
                name: 'base',
                type: 'number',
                declaringStmt: {
                    text: 'base = %closures0.base'
                },
            },
            {
                name: 'input',
                type: 'number',
                declaringStmt: {
                    text: 'input = parameter1: number'
                },
            }
        ]
    }
};

export const BasicOuterMethod4_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: %dflt.basicOuterMethod4()`,
        methodSubSignature: {
            returnType: 'closures: @function/ClosureParamsTest.ts: %dflt.%basicNestedMethod4$basicOuterMethod4([base], number)'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `%closures0`,
                type: '[base]',
                declaringStmt: null,
                usedStmts: []
            },
            {
                name: 'base',
                type: 'number',
                declaringStmt: {
                    text: `base = 3`
                },
                usedStmts: []
            },
            {
                name: '%basicNestedMethod4$basicOuterMethod4',
                type: 'closures: @function/ClosureParamsTest.ts: %dflt.%basicNestedMethod4$basicOuterMethod4([base], number)',
                declaringStmt: null,
                usedStmts: [
                    {
                        text: 'return %basicNestedMethod4$basicOuterMethod4'
                    }
                ]
            }
        ],
        stmts: [
            {
                text: 'this = this: @function/ClosureParamsTest.ts: %dflt',
            },
            {
                text: 'base = 3',
            },
            {
                text: 'return %basicNestedMethod4$basicOuterMethod4',
            }
        ]
    }
};

export const CallMethod4_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: %dflt.callMethod4()`,
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: 'callMethod',
                type: 'closures: @function/ClosureParamsTest.ts: %dflt.%basicNestedMethod4$basicOuterMethod4([base], number)',
                declaringStmt: {
                    text: 'callMethod = staticinvoke <@function/ClosureParamsTest.ts: %dflt.basicOuterMethod4()>()'
                },
                usedStmts: [
                    {
                        // TODO: 此处需要类型推导时进行处理
                        text: 'staticinvoke <@%unk/%unk: .callMethod()>(3)'
                    }
                ]
            }
        ],
        globals: [
            {
                name: 'basicOuterMethod4',
                instanceof: GlobalRef,
                // TODO: 此处需要类型推导时进行处理
                ref: null,
                usedStmts: [
                    {
                        text: 'callMethod = staticinvoke <@function/ClosureParamsTest.ts: %dflt.basicOuterMethod4()>()'
                    }
                ]
            }
        ],
        stmts: [
            {
                text: 'this = this: @function/ClosureParamsTest.ts: %dflt',
            },
            {
                text: 'callMethod = staticinvoke <@function/ClosureParamsTest.ts: %dflt.basicOuterMethod4()>()',
            },
            {
                text: 'staticinvoke <@%unk/%unk: .callMethod()>(3)',
            },
            {
                text: 'return',
            }
        ]
    }
};

export const UnClosureFunction_Expect_IR = {
    outerMethod: undefined,
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.outerFunction1(number)`,
        methodSubSignature: {
            returnType: 'void'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: 'this',
                type: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}`,
            },
            {
                name: 'outerInput',
                type: 'number',
            },
            {
                name: 'count',
                type: 'number',
            },
            {
                name: 'flag',
                type: 'number',
            },
            {
                name: 'innerFunction2',
                type: `closures: @function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${ANONYMOUS_METHOD_PREFIX}1${NAME_DELIMITER}outerFunction1([outerInput])`,
                declaringStmt: {
                    text: `innerFunction2 = ${ANONYMOUS_METHOD_PREFIX}1${NAME_DELIMITER}outerFunction1`
                },
                usedStmts: [
                    {
                        text: `ptrinvoke <@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${ANONYMOUS_METHOD_PREFIX}1${NAME_DELIMITER}outerFunction1([outerInput])>(${LEXICAL_ENV_NAME_PREFIX}1)`
                    }
                ]
            },
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                type: `[count, flag]`,
                usedStmts: [
                    {
                        text: `%0 = staticinvoke <@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction1${NAME_DELIMITER}outerFunction1([count, flag], string)>(${LEXICAL_ENV_NAME_PREFIX}0, \'abc\')`
                    }
                ]
            },
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}1`,
                type: `[outerInput]`,
                usedStmts: [
                    {
                        text: `ptrinvoke <@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.%AM1$outerFunction1([outerInput])>(${LEXICAL_ENV_NAME_PREFIX}1)`
                    }
                ]
            }
        ],
        globals: [
            {
                name: `console`,
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'instanceinvoke console.<@%unk/%unk: .log()>(%0)'
                    }
                ]
            },
            {
                name: `innerFunction1`,
                instanceof: GlobalRef,
                ref: {
                    instanceof: ArkStaticFieldRef,
                    declaringSignature: '@function/ClosureParamsTest.ts: %dflt',
                    fieldName: '%innerFunction1$outerFunction1',
                    type: `closures: @function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction1${NAME_DELIMITER}outerFunction1([count, flag], string)`,
                },
                usedStmts: [
                    {
                        text: `%0 = staticinvoke <@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction1${NAME_DELIMITER}outerFunction1([count, flag], string)>(${LEXICAL_ENV_NAME_PREFIX}0, \'abc\')`
                    }
                ]
            }
        ]
    }
};

export const ClosureFunction_Expect_IR = {
    outerMethod: {
        toString: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.outerFunction1(number)`,
    },
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction1${NAME_DELIMITER}outerFunction1([count, flag], string)`,
        methodSubSignature: {
            parameters: [
                {
                    name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                    type: '[count, flag]'
                },
                {
                    name: 'innerInput',
                    type: 'string'
                }
            ],
            returnType: 'string'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                type: '[count, flag]',
                declaringStmt: {
                    text: `${LEXICAL_ENV_NAME_PREFIX}0 = parameter0: [count, flag]`
                },
                usedStmts: [
                    {
                        text: `count = ${LEXICAL_ENV_NAME_PREFIX}0.count`
                    },
                    {
                        text: `flag = ${LEXICAL_ENV_NAME_PREFIX}0.flag`
                    }
                ]
            },
            {
                name: 'count',
                type: 'number',
                declaringStmt: {
                    text: `count = ${LEXICAL_ENV_NAME_PREFIX}0.count`
                },
                usedStmts: [
                    {
                        text: 'count = count + 1'
                    }
                ]
            },
            {
                name: 'flag',
                type: 'number',
                declaringStmt: {
                    text: `flag = ${LEXICAL_ENV_NAME_PREFIX}0.flag`
                },
                usedStmts: [
                    {
                        text: 'if flag == 1'
                    },
                    {
                        text: 'if flag == 2'
                    }
                ]
            },
            {
                name: 'innerInput',
                type: 'string',
                declaringStmt: {
                    text: 'innerInput = parameter1: string'
                },
            },
            {
                name: 'this',
                type: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}`,
            },
            {
                name: 'result',
                type: 'string',
            },
        ],
        globals: undefined
    }
};

export const ClosureAnonymousFunction_Expect_IR = {
    outerMethod: {
        toString: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.outerFunction1(number)`,
    },
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}.${ANONYMOUS_METHOD_PREFIX}1${NAME_DELIMITER}outerFunction1([outerInput])`,
        methodSubSignature: {
            parameters: [
                {
                    name: `${LEXICAL_ENV_NAME_PREFIX}1`,
                    type: '[outerInput]'
                }
            ],
            returnType: 'void'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}1`,
                type: '[outerInput]',
                declaringStmt: {
                    text: `${LEXICAL_ENV_NAME_PREFIX}1 = parameter0: [outerInput]`
                },
                usedStmts: [
                    {
                        text: `outerInput = ${LEXICAL_ENV_NAME_PREFIX}1.outerInput`
                    }
                ]
            },
            {
                name: 'outerInput',
                type: 'number',
                declaringStmt: {
                    text: `outerInput = ${LEXICAL_ENV_NAME_PREFIX}1.outerInput`
                },
            },
            {
                name: 'this',
                type: `@function/ClosureParamsTest.ts: ${DEFAULT_ARK_CLASS_NAME}`,
            }
        ],
        globals: [
            {
                name: 'console',
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'instanceinvoke console.<@%unk/%unk: .log()>(outerInput)'
                    }
                ]
            }
        ]
    }
};

export const ClosureClassMethod_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: ClosureClass.outerFunction2(number)'
    },
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: ClosureClass.${NAME_PREFIX}innerFunction2${NAME_DELIMITER}outerFunction2([count, nums], string)`,
        methodSubSignature: {
            parameters: [
                {
                    name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                    type: '[count, nums]'
                },
                {
                    name: 'outerInput',
                    type: 'string'
                }
            ],
            returnType: 'string'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                type: '[count, nums]',
                declaringStmt: {
                    text: `${LEXICAL_ENV_NAME_PREFIX}0 = parameter0: [count, nums]`
                },
                usedStmts: [
                    {
                        text: `count = ${LEXICAL_ENV_NAME_PREFIX}0.count`
                    },
                    {
                        text: `nums = ${LEXICAL_ENV_NAME_PREFIX}0.nums`
                    },
                ]
            },
            {
                name: 'count',
                type: 'string',
                declaringStmt: {
                    text: `count = ${LEXICAL_ENV_NAME_PREFIX}0.count`
                },
                usedStmts: [
                    {
                        text: 'count = count + outerInput'
                    },
                    {
                        text: 'count = count + item'
                    },
                    {
                        text: '%5 = %4 + count'
                    }
                ]
            },
            {
                name: 'nums',
                type: 'number[]',
                declaringStmt: {
                    text: `nums = ${LEXICAL_ENV_NAME_PREFIX}0.nums`
                },
                usedStmts: [
                    {
                        text: '%0 = instanceinvoke nums.<@%unk/%unk: .iterator()>()'
                    }
                ]
            },
            {
                name: 'outerInput',
                type: 'string',
                declaringStmt: {
                    text: 'outerInput = parameter1: string'
                }
            },
            {
                name: 'this',
                type: '@function/ClosureParamsTest.ts: ClosureClass',
            },
            {
                name: 'item',
                type: 'number',
            }
        ],
    }
};

export const ClosureNamespaceFunction_Expect_IR = {
    outerMethod: {
        toString: `@function/ClosureParamsTest.ts: closureNamespace.${DEFAULT_ARK_CLASS_NAME}.outerFunction3(number)`
    },
    methodDeclareSignatures: [
        {
            toString: `@function/ClosureParamsTest.ts: closureNamespace.${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction3${NAME_DELIMITER}outerFunction3([count, size, outerInput])`,
            methodSubSignature: {
                returnType: 'string'
            }
        },
        {
            toString: `@function/ClosureParamsTest.ts: closureNamespace.${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction3${NAME_DELIMITER}outerFunction3([count, size, outerInput], string)`,
            methodSubSignature: {
                returnType: 'string'
            }
        }

    ],
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: closureNamespace.${DEFAULT_ARK_CLASS_NAME}.${NAME_PREFIX}innerFunction3${NAME_DELIMITER}outerFunction3([count, size, outerInput], string)`,
        methodSubSignature: {
            parameters: [
                {
                    name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                    type: '[count, size, outerInput]'
                },
                {
                    name: 'innerInput',
                    type: 'string'
                }
            ],
            returnType: 'string'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                type: '[count, size, outerInput]',
                declaringStmt: {
                    text: `${LEXICAL_ENV_NAME_PREFIX}0 = parameter0: [count, size, outerInput]`
                },
                usedStmts: [
                    {
                        text: `count = ${LEXICAL_ENV_NAME_PREFIX}0.count`
                    },
                    {
                        text: `size = ${LEXICAL_ENV_NAME_PREFIX}0.size`
                    },
                    {
                        text: `outerInput = ${LEXICAL_ENV_NAME_PREFIX}0.outerInput`
                    }
                ]
            },
            {
                name: 'count',
                type: 'number',
                declaringStmt: {
                    text: `count = ${LEXICAL_ENV_NAME_PREFIX}0.count`
                },
                usedStmts: [
                    {
                        text: '%0 = count + size'
                    }
                ]
            },
            {
                name: 'size',
                type: 'number',
                declaringStmt: {
                    text: `size = ${LEXICAL_ENV_NAME_PREFIX}0.size`
                },
                usedStmts: [
                    {
                        text: '%0 = count + size'
                    }
                ]
            },
            {
                name: 'outerInput',
                type: 'number',
                declaringStmt: {
                    text: `outerInput = ${LEXICAL_ENV_NAME_PREFIX}0.outerInput`
                },
                usedStmts: [
                    {
                        text: '%1 = outerInput + \': \''
                    }
                ]
            },
            {
                name: 'innerInput',
                type: 'string',
                declaringStmt: {
                    text: 'innerInput = parameter1: string'
                },
                usedStmts: []
            },
            {
                name: 'this',
                type: `@function/ClosureParamsTest.ts: closureNamespace.${DEFAULT_ARK_CLASS_NAME}`,
            },
            {
                name: 'res',
                type: 'unknown',
            }
        ],
        globals: [
            {
                name: 'globalValue',
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'res = %0 + globalValue'
                    }
                ]
            }
        ]
    }
};

export const ClosureNamespaceClassMethod_Expect_IR = {
    outerMethod: {
        toString: '@function/ClosureParamsTest.ts: closureNamespace.ClosureClass.outerFunction3(number)'
    },
    methodSignature: {
        toString: `@function/ClosureParamsTest.ts: closureNamespace.ClosureClass.${NAME_PREFIX}innerFunction3${NAME_DELIMITER}outerFunction3([flag, outerInput])`,
        methodSubSignature: {
            parameters: [
                {
                    name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                    type: '[flag, outerInput]'
                }
            ],
            returnType: 'void'
        }
    },
    bodyBuilder: undefined,
    body: {
        locals: [
            {
                name: `${LEXICAL_ENV_NAME_PREFIX}0`,
                type: '[flag, outerInput]',
                declaringStmt: {
                    text: `${LEXICAL_ENV_NAME_PREFIX}0 = parameter0: [flag, outerInput]`
                },
                usedStmts: [
                    {
                        text: `flag = ${LEXICAL_ENV_NAME_PREFIX}0.flag`
                    },
                    {
                        text: `outerInput = ${LEXICAL_ENV_NAME_PREFIX}0.outerInput`
                    }
                ]
            },
            {
                name: 'flag',
                type: 'boolean',
                declaringStmt: {
                    text: `flag = ${LEXICAL_ENV_NAME_PREFIX}0.flag`
                },
                usedStmts: [
                    {
                        text: '%0 = !flag',
                    }
                ]
            },
            {
                name: 'outerInput',
                type: 'number',
                declaringStmt: {
                    text: `outerInput = ${LEXICAL_ENV_NAME_PREFIX}0.outerInput`
                },
                usedStmts: [
                    {
                        text: 'if outerInput > 0'
                    },
                    {
                        text: 'outerInput = outerInput - 1'
                    }
                ]
            },
            {
                name: 'this',
                type: '@function/ClosureParamsTest.ts: closureNamespace.ClosureClass'
            },
            {
                name: `error`,
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'instanceinvoke console.<@%unk/%unk: .log()>(error)'
                    }
                ]
            }
        ],
        globals: [
            {
                name: `console`,
                type: 'unknown',
                instanceof: GlobalRef,
                usedStmts: [
                    {
                        text: 'instanceinvoke console.<@%unk/%unk: .log()>(error)'
                    }
                ]
            },
        ]
    }
};