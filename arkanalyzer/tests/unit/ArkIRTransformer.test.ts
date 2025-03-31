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

import { assert, describe, expect, it } from 'vitest';
import path from 'path';
import {
    ANONYMOUS_METHOD_PREFIX,
    ArkMethod,
    DEFAULT_ARK_CLASS_NAME,
    ArkStaticFieldRef,
    DEFAULT_ARK_METHOD_NAME,
    GlobalRef,
    Local,
    ModelUtils,
    NAME_DELIMITER,
    NAME_PREFIX,
    Scene,
    SceneConfig,
    Stmt,
    Value,
} from '../../src';
import {
    BinaryExpression_Expect_IR,
    CallExpression_Expect_IR,
    LiteralExpression_Expect_IR,
    NewExpression_Expect_IR,
    Operator_Expect_IR,
    UnaryExpression_Expect_IR,
} from '../resources/arkIRTransformer/expression/ExpressionExpectIR';
import {
    CompoundAssignment_Expect_IR,
    Declaration_Expect_IR, Destructuring_Expect_IR,
} from '../resources/arkIRTransformer/assignment/AssignmentExpectIR';
import {
    ArrowFunction_Expect_IR,
    BasicNestedMethod1_Expect_IR,
    BasicNestedMethod2_Expect_IR,
    BasicNestedMethod3_Expect_IR,
    BasicNestedMethod4_Expect_IR,
    BasicNestedMethod_Expect_IR,
    BasicOuterMethod1_Expect_IR,
    BasicOuterMethod2_Expect_IR,
    BasicOuterMethod3_Expect_IR,
    BasicOuterMethod4_Expect_IR,
    BasicOuterMethod_Expect_IR,
    CallMethod4_Expect_IR,
    ClosureAnonymousFunction_Expect_IR,
    ClosureClassMethod_Expect_IR,
    ClosureFunction_Expect_IR,
    ClosureNamespaceClassMethod_Expect_IR,
    ClosureNamespaceFunction_Expect_IR,
    NoOverloadMethod_Expect_IR,
    NoOverloadMethodWithBody2_Expect_IR,
    NoOverloadMethodWithBody_Expect_IR,
    OverloadClassMethod_Expect_IR,
    OverloadInterfaceMethod_Expect_IR,
    OverloadMethod_Expect_IR,
    OverloadNamespaceMethod_Expect_IR,
    UnClosureFunction_Expect_IR,
} from '../resources/arkIRTransformer/function/FunctionExpectIR';
import { MethodParameter } from '../../src/core/model/builder/ArkMethodBuilder';

const BASE_DIR = path.join(__dirname, '../../tests/resources/arkIRTransformer');

export function testMethodStmts(scene: Scene, fileName: string, expectStmts: any[],
                                className: string = DEFAULT_ARK_CLASS_NAME,
                                methodName: string = DEFAULT_ARK_METHOD_NAME): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(fileName));
    const arkMethod = arkFile?.getClassWithName(className)?.getMethods()
        .find((method) => (method.getName() === methodName));
    const stmts = arkMethod?.getCfg()?.getStmts();
    if (!stmts) {
        assert.isDefined(stmts);
        return;
    }
    assertStmtsEqual(stmts, expectStmts);
}

function testFileStmts(scene: Scene, filePath: string, expectFileStmts: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    if (!arkFile) {
        assert.isDefined(arkFile);
        return;
    }
    const methods = ModelUtils.getAllMethodsInFile(arkFile);
    for (const expectMethod of expectFileStmts.methods) {
        const expectMethodName = expectMethod.name;
        const method = methods.find((method) => method.getName() === expectMethodName);
        if (!method) {
            assert.isDefined(method);
            continue;
        }
        const stmts = method.getCfg()?.getStmts();
        if (!stmts) {
            assert.isDefined(stmts);
            continue;
        }
        assertStmtsEqual(stmts, expectMethod.stmts);
    }
}

function assertStmtsEqual(stmts: Stmt[], expectStmts: any[]): void {
    expect(stmts.length).toEqual(expectStmts.length);
    for (let i = 0; i < stmts.length; i++) {
        expect(stmts[i].toString()).toEqual(expectStmts[i].text);

        if (expectStmts[i].operandOriginalPositions === undefined) {
            continue;
        }
        const operandOriginalPositions: any[] = [];
        for (const operand of stmts[i].getDefAndUses()) {
            const operandOriginalPosition = stmts[i].getOperandOriginalPosition(operand);
            if (operandOriginalPosition) {
                operandOriginalPositions.push(
                    [operandOriginalPosition.getFirstLine(), operandOriginalPosition.getFirstCol(),
                        operandOriginalPosition.getLastLine(), operandOriginalPosition.getLastCol()]);
            } else {
                operandOriginalPositions.push(operandOriginalPosition);
            }
        }
        expect(operandOriginalPositions).toEqual(expectStmts[i].operandOriginalPositions);
    }
}

function testMethodOverload(scene: Scene, filePath: string, methodName: string, expectMethod: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
    if (arkMethod === undefined) {
        assert.isDefined(arkMethod);
        return;
    }
    assertMethodLineEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, true, expectMethod);
    assertMethodSignatureEqual(arkMethod, expectMethod);
}

function testClassMethodOverload(scene: Scene, filePath: string, className: string, methodName: string, expectMethod: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkClass = arkFile?.getClasses().find((cls) => cls.getName() === className);
    const arkMethod = arkClass?.getMethods().find((method) => (method.getName() === methodName));
    if (arkMethod === undefined) {
        assert.isDefined(arkMethod);
        return;
    }
    assertMethodLineEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, true, expectMethod);
    assertMethodSignatureEqual(arkMethod, expectMethod);
}

function testNamespaceMethodOverload(scene: Scene, filePath: string, namespaceName: string, methodName: string, expectMethod: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkNamespace = arkFile?.getNamespaces().find((ns) => ns.getName() === namespaceName);
    const arkMethod = arkNamespace?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
    if (arkMethod === undefined) {
        assert.isDefined(arkMethod);
        return;
    }
    assertMethodLineEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, false);
    assertMethodSignatureEqual(arkMethod, expectMethod);
}

function testInterfaceMethodOverload(scene: Scene, filePath: string, interfaceName: string, methodName: string, expectMethod: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkClass = arkFile?.getClasses().find((cls) => cls.getName() === interfaceName);
    const arkMethod = arkClass?.getMethods().find((method) => (method.getName() === methodName));
    if (arkMethod === undefined) {
        assert.isDefined(arkMethod);
        return;
    }
    assertMethodLineEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, false, expectMethod);
    assertMethodSignatureEqual(arkMethod, expectMethod);
}

function testNoMethodOverload(scene: Scene, filePath: string, methodName: string, expectMethod: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
    if (arkMethod === undefined) {
        assert.isDefined(arkMethod);
        return;
    }
    assertMethodLineEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, false);
    assertMethodSignatureEqual(arkMethod, expectMethod);
}

function testNoMethodOverloadWithBody(scene: Scene, filePath: string, methodName: string, expectMethod: any): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
    if (arkMethod === undefined) {
        assert.isDefined(arkMethod);
        return;
    }
    assertMethodLineEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, true, expectMethod);
    assertMethodSignatureEqual(arkMethod, expectMethod);
}

function assertMethodLineEqual(method: ArkMethod, expectMethod: any): void {
    const declareLines = method.getDeclareLines();
    const expectDeclareLines = expectMethod.methodDeclareLines;
    if (expectDeclareLines !== undefined) {
        if (expectDeclareLines === null) {
            assert.isNull(declareLines);
        } else {
            assert.isNotNull(declareLines);
            expect(declareLines?.length).toEqual(expectDeclareLines.length);
            declareLines?.forEach((line, index) => {
                expect(line).toEqual(expectDeclareLines[index]);
            });
        }
    }

    if (expectMethod.line !== undefined) {
        if (expectMethod.line === null) {
            assert.isNull(method.getLine());
        } else {
            expect(method.getLine()).toEqual(expectMethod.line);
        }
    }
}

function assertLocalsEqual(actualLocals: Map<string, Local>, expectLocals: any): void {
    for (let i = 0; i < expectLocals.length; i++) {
        const actualLocal = actualLocals.get(expectLocals[i].name);
        assert.isDefined(actualLocal);
        if (expectLocals[i].type !== undefined) {
            assert.equal(actualLocal!.getType().toString(), expectLocals[i].type);
        }
        if (expectLocals[i].declaringStmt !== undefined) {
            if (expectLocals[i].declaringStmt === null) {
                expect(actualLocal!.getDeclaringStmt()).toEqual(expectLocals[i].declaringStmt);
            } else {
                expect(actualLocal!.getDeclaringStmt()?.toString()).toEqual(expectLocals[i].declaringStmt.text);
            }
        }
        if (expectLocals[i].usedStmts !== undefined) {
            assertStmtsEqual(actualLocal!.getUsedStmts(), expectLocals[i].usedStmts);
        }
    }
}

function assertGlobalsEqual(actualGlobals: Map<string, Value> | undefined, expectGlobals: any): void {
    if (expectGlobals !== undefined && expectGlobals !== null) {
        assert.isDefined(actualGlobals);
    } else {
        assert.isUndefined(actualGlobals);
        return;
    }
    for (let i = 0; i < expectGlobals.length; i++) {
        const actualGlobal = actualGlobals!.get(expectGlobals[i].name);
        assert.isDefined(actualGlobal);
        if (expectGlobals[i].instanceof !== undefined) {
            assert.isTrue(actualGlobal instanceof expectGlobals[i].instanceof);
            if (expectGlobals[i].ref === null) {
                const refValue = (actualGlobal as GlobalRef).getRef();
                assert.isNull(refValue);
            } else if (expectGlobals[i].ref !== undefined && expectGlobals[i].instanceof === GlobalRef) {
                const refValue = (actualGlobal as GlobalRef).getRef();
                assert.isNotNull(refValue);
                assertGlobalRefEqual(refValue!, expectGlobals[i].ref);
            }
        }
        if (expectGlobals[i].type !== undefined) {
            assert.equal(actualGlobal!.getType().toString(), expectGlobals[i].type);
        }
        if (expectGlobals[i].usedStmts !== undefined) {
            if (actualGlobal instanceof GlobalRef) {
                assertStmtsEqual(actualGlobal!.getUsedStmts(), expectGlobals[i].usedStmts);
            }
        }
    }
}

function assertGlobalRefEqual(refValue: Value, expectedRef: any): void {
    if (expectedRef.type !== undefined) {
        assert.equal(refValue.getType().toString(), expectedRef.type);
    }
    if (expectedRef.instanceof !== undefined) {
        assert.isTrue(refValue instanceof expectedRef.instanceof);
        if (expectedRef.instanceof === ArkStaticFieldRef) {
            const fieldSignature = (refValue as ArkStaticFieldRef).getFieldSignature();
            if (expectedRef.fieldName !== undefined) {
                assert.equal(fieldSignature.getFieldName(), expectedRef.fieldName);
            }
            if (expectedRef.declaringSignature !== undefined) {
                assert.equal(fieldSignature.getDeclaringSignature().toString(), expectedRef.declaringSignature);
            }
        }
    }
}

function assertMethodBodyEqual(method: ArkMethod, expectBodyDefined: boolean, expectMethod?: any): void {
    const body = method.getBody();
    if (!expectBodyDefined) {
        assert.isUndefined(body);
        return;
    }
    assert.isDefined(expectMethod);
    if (expectMethod.body !== undefined) {
        assert.isDefined(body);
        if (expectMethod.body.locals !== undefined) {
            assertLocalsEqual(body!.getLocals(), expectMethod.body.locals);
        }
        assertGlobalsEqual(body!.getUsedGlobals(), expectMethod.body.globals);
        if (expectMethod.body.stmts !== undefined) {
            assertStmtsEqual(body!.getCfg().getStmts(), expectMethod.body.stmts);
        }
    }
}

function assertMethodBodyBuilderEqual(method: ArkMethod, expectedBodyBuilder: any): void {
    const bodyBuilder = method.getBodyBuilder();
    if (expectedBodyBuilder === undefined) {
        assert.isUndefined(bodyBuilder);
    } else {
        assert.isDefined(bodyBuilder);
    }
}

function assertParamsEqual(actualParams: MethodParameter[], expectedParams: any): void {
    if (expectedParams !== undefined) {
        assert.equal(actualParams.length, expectedParams.length);
        for (let i = 0; i < expectedParams.length; i++) {
            assert.equal(actualParams[i].getName(), expectedParams[i].name);
            assert.equal(actualParams[i].getType().getTypeString(), expectedParams[i].type);
        }
    }
}

function assertMethodSignatureEqual(method: ArkMethod, expectMethod?: any): void {
    const declareSignatures = method.getDeclareSignatures();
    const expectDeclareSignatures = expectMethod.methodDeclareSignatures;
    if (expectDeclareSignatures === null) {
            assert.isNull(declareSignatures);
    } else if (expectDeclareSignatures !== undefined) {
        assert.isNotNull(declareSignatures);
        assert.equal(declareSignatures?.length, expectDeclareSignatures.length);
        let index = 0;
        for (let signature of declareSignatures!) {
            assert.equal(signature.toString(), expectDeclareSignatures[index].toString);
            const expectDeclareSubSignatures = expectDeclareSignatures[index].methodSubSignature;
            index++;
            if (expectDeclareSubSignatures === undefined) {
                continue;
            }
            if (expectDeclareSubSignatures.returnType !== undefined) {
                assert.equal(signature.getType().toString(), expectDeclareSubSignatures.returnType);
            }
            if (expectDeclareSubSignatures.parameters !== undefined) {
                assertParamsEqual(signature.getMethodSubSignature().getParameters(), expectDeclareSubSignatures.parameters);
            }
        }
    }

    const implementationSignature = method.getImplementationSignature();
    const expectImplSignature = expectMethod.methodSignature;
    if (expectImplSignature !== undefined && expectImplSignature !== null) {
        assert.isNotNull(implementationSignature);
        assert.equal(implementationSignature!.toString(), expectImplSignature.toString);
        const expectImplSubSignature = expectImplSignature.methodSubSignature;
        if (expectImplSubSignature !== undefined) {
            if (expectImplSubSignature.returnType !== undefined) {
                assert.equal(implementationSignature!.getType().toString(), expectImplSubSignature.returnType);
            }
            if (expectImplSubSignature.parameters !== undefined) {
                assertParamsEqual(implementationSignature!.getMethodSubSignature().getParameters(), expectImplSubSignature.parameters);
            }
        }
    }
}

function assertOuterMethodEqual(method: ArkMethod, outerMethod: any): void {
    if (outerMethod === undefined || outerMethod === null || outerMethod.toString === undefined) {
        assert.isUndefined(method.getOuterMethod());
        return;
    }
    assert.equal(method.getOuterMethod()?.getSignature().toString(), outerMethod.toString);
}

function testMethodClosure(arkMethod: ArkMethod, expectMethod: any): void {
    assertOuterMethodEqual(arkMethod, expectMethod.outerMethod);
    assertMethodSignatureEqual(arkMethod, expectMethod);
    assertMethodBodyEqual(arkMethod, true, expectMethod);
    assertMethodBodyBuilderEqual(arkMethod, expectMethod.bodyBuilder);
}

function buildScene(folderName: string) {
    let config: SceneConfig = new SceneConfig();
    config.buildFromProjectDir(path.join(BASE_DIR, folderName));
    let scene = new Scene();
    scene.buildSceneFromProjectDir(config);
    scene.inferTypes();
    return scene;
}

describe('expression Test', () => {
    const scene = buildScene('expression');

    it('test binary expression', async () => {
        testMethodStmts(scene, 'BinaryExpressionTest.ts', BinaryExpression_Expect_IR.stmts);
    });

    it('test unary expression', async () => {
        testMethodStmts(scene, 'UnaryExpressionTest.ts', UnaryExpression_Expect_IR.stmts);
    });

    it('test new expression', async () => {
        testMethodStmts(scene, 'NewExpressionTest.ts', NewExpression_Expect_IR.stmts);
    });

    it('test literal expression', async () => {
        testMethodStmts(scene, 'LiteralExpressionTest.ts', LiteralExpression_Expect_IR.stmts);
    });

    it('test operator', async () => {
        testMethodStmts(scene, 'OperatorTest.ts', Operator_Expect_IR.stmts);
    });

    it('test call expression', async () => {
        testMethodStmts(scene, 'CallExpressionTest.ts', CallExpression_Expect_IR.stmts);
    });
});

describe('assignment Test', () => {
    const scene = buildScene('assignment');

    it('test declaration', async () => {
        testMethodStmts(scene, 'DeclarationTest.ts', Declaration_Expect_IR.stmts);
    });

    it('test compound assignment', async () => {
        testMethodStmts(scene, 'CompoundAssignmentTest.ts', CompoundAssignment_Expect_IR.stmts);
    });

    it('test destructuring', async () => {
        testMethodStmts(scene, 'DestructuringSample.ts', Destructuring_Expect_IR.stmts);
    });
});

describe('function Test', () => {
    const scene = buildScene('function');

    it('test arrow function', async () => {
        testFileStmts(scene, 'ArrowFunctionTest.ts', ArrowFunction_Expect_IR);
    });

    it('test overload function', async () => {
        testMethodOverload(scene, 'OverloadFunctionTest.ts', 'overloadedFunction1', OverloadMethod_Expect_IR);
    });

    it('test overload class function', async () => {
        testClassMethodOverload(scene, 'OverloadFunctionTest.ts', 'OverloadClass', 'overloadedFunction2', OverloadClassMethod_Expect_IR);
    });

    it('test overload namespace function', async () => {
        testNamespaceMethodOverload(scene, 'OverloadFunctionTest.ts', 'overloadNamespace', 'overloadedFunction3', OverloadNamespaceMethod_Expect_IR);
    });

    it('test overload interface function', async () => {
        testInterfaceMethodOverload(scene, 'OverloadFunctionTest.ts', 'OverloadInterface', 'overloadedFunction4', OverloadInterfaceMethod_Expect_IR);
    });

    it('test no overload function without body', async () => {
        testNoMethodOverload(scene, 'OverloadFunctionTest.ts', 'function5', NoOverloadMethod_Expect_IR);
    });

    it('test no overload function with body', async () => {
        testNoMethodOverloadWithBody(scene, 'OverloadFunctionTest.ts', 'function6', NoOverloadMethodWithBody_Expect_IR);
    });

    it('test no overload function with body 2', async () => {
        testNoMethodOverloadWithBody(scene, 'OverloadFunctionTest.ts', 'function7', NoOverloadMethodWithBody2_Expect_IR);
    });
});

describe('closure Test', () => {
    const scene = buildScene('function');
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith('ClosureParamsTest.ts'));

    it('basic test nested function with no closures', async () => {
        const nestedMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === '%nestedMethod$basicMethod'));
        assert.isDefined(nestedMethod);
        testMethodClosure(nestedMethod as ArkMethod, BasicNestedMethod_Expect_IR);

        const outerMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === 'basicMethod'));
        assert.isDefined(outerMethod);
        testMethodClosure(outerMethod as ArkMethod, BasicOuterMethod_Expect_IR);
    });

    it('basic test independently nested function declaration', async () => {
        const nestedMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === '%basicNestedMethod1$basicOuterMethod1'));
        assert.isDefined(nestedMethod);
        testMethodClosure(nestedMethod as ArkMethod, BasicNestedMethod1_Expect_IR);

        const outerMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === 'basicOuterMethod1'));
        assert.isDefined(outerMethod);
        testMethodClosure(outerMethod as ArkMethod, BasicOuterMethod1_Expect_IR);
    });

    it('basic test anonymous nested function declared in forEach', async () => {
        const nestedMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === '%AM0$basicOuterMethod2'));
        assert.isDefined(nestedMethod);
        testMethodClosure(nestedMethod!, BasicNestedMethod2_Expect_IR);

        const outerMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === 'basicOuterMethod2'));
        assert.isDefined(outerMethod);
        testMethodClosure(outerMethod!, BasicOuterMethod2_Expect_IR);
    });

    it('basic test ptr invoke anonymous nested function', async () => {
        const nestedMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === '%AM1$basicOuterMethod3'));
        assert.isDefined(nestedMethod);
        testMethodClosure(nestedMethod!, BasicNestedMethod3_Expect_IR);

        const outerMethod = arkFile?.getClassWithName('BasicTest')?.getMethods().find((method) => (method.getName() === 'basicOuterMethod3'));
        assert.isDefined(outerMethod);
        testMethodClosure(outerMethod!, BasicOuterMethod3_Expect_IR);
    });

    it('basic test return nested function', async () => {
        const nestedMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === '%basicNestedMethod4$basicOuterMethod4'));
        assert.isDefined(nestedMethod);
        testMethodClosure(nestedMethod!, BasicNestedMethod4_Expect_IR);

        const outerMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === 'basicOuterMethod4'));
        assert.isDefined(outerMethod);
        testMethodClosure(outerMethod!, BasicOuterMethod4_Expect_IR);

        const callMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === 'callMethod4'));
        assert.isDefined(callMethod);
        testMethodClosure(callMethod!, CallMethod4_Expect_IR);
    });

    it('test closure in function', async () => {
        const methodName = `${NAME_PREFIX}innerFunction1${NAME_DELIMITER}outerFunction1`;
        const arkMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
        assert.isDefined(arkMethod);
        testMethodClosure(arkMethod as ArkMethod, ClosureFunction_Expect_IR);
    });

    it('test unClosure function', async () => {
        const arkMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === 'outerFunction1'));
        assert.isDefined(arkMethod);
        testMethodClosure(arkMethod as ArkMethod, UnClosureFunction_Expect_IR);
    });

    it('test closure in anonymous function', async () => {
        const methodName = `${ANONYMOUS_METHOD_PREFIX}1${NAME_DELIMITER}outerFunction1`;
        const arkMethod = arkFile?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
        assert.isDefined(arkMethod);
        testMethodClosure(arkMethod as ArkMethod, ClosureAnonymousFunction_Expect_IR);
    });

    it('test closure in class method', async () => {
        const arkClass = arkFile?.getClasses().find((cls) => (cls.getName() === 'ClosureClass'));
        const methodName = `${NAME_PREFIX}innerFunction2${NAME_DELIMITER}outerFunction2`;
        const arkMethod = arkClass?.getMethods().find((method) => (method.getName() === methodName));
        assert.isDefined(arkMethod);
        testMethodClosure(arkMethod as ArkMethod, ClosureClassMethod_Expect_IR);
    });

    it('test closure in Namespace function', async () => {
        const arkNS = arkFile?.getNamespaces().find((ns) => ns.getName() === 'closureNamespace');
        const methodName = `${NAME_PREFIX}innerFunction3${NAME_DELIMITER}outerFunction3`;
        const arkMethod = arkNS?.getDefaultClass().getMethods().find((method) => (method.getName() === methodName));
        assert.isDefined(arkMethod);
        testMethodClosure(arkMethod as ArkMethod, ClosureNamespaceFunction_Expect_IR);
    });

    it('test closure Namespace Class method', async () => {
        const arkNS = arkFile?.getNamespaces().find((ns) => ns.getName() === 'closureNamespace');
        const methodName = `${NAME_PREFIX}innerFunction3${NAME_DELIMITER}outerFunction3`;
        const arkMethod = arkNS?.getClassWithName('ClosureClass')?.getMethods().find((method) => (method.getName() === methodName));
        assert.isDefined(arkMethod);
        testMethodClosure(arkMethod as ArkMethod, ClosureNamespaceClassMethod_Expect_IR);
    });
});