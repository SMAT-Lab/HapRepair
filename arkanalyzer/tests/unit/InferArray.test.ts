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

import { assert, describe, it } from 'vitest';
import path from 'path';
import { SceneConfig } from '../../src/Config';
import { Scene } from '../../src/Scene';
import { FileSignature } from '../../src/core/model/ArkSignature';
import {
    AliasType,
    ArkAssignStmt,
    ArkInstanceFieldRef,
    ArkInvokeStmt,
    ArkNewArrayExpr,
    ArkStaticFieldRef,
    ArrayType,
    ClassType,
    NumberType,
    StringType,
    UnionType,
} from '../../src';
import Logger, { LOG_LEVEL, LOG_MODULE_TYPE } from '../../src/utils/logger';

const logPath = 'out/ArkAnalyzer.log';
const logger = Logger.getLogger(LOG_MODULE_TYPE.TOOL, 'InferArrayTest');
Logger.configure(logPath, LOG_LEVEL.DEBUG, LOG_LEVEL.DEBUG);

describe("Infer Array Test", () => {

    let config: SceneConfig = new SceneConfig();
    config.buildFromProjectDir(path.join(__dirname, "../resources/inferType"));
    let projectScene: Scene = new Scene();
    projectScene.buildSceneFromProjectDir(config);
    projectScene.inferTypes();
    it('normal case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'inferSample.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getDefaultClass().getMethodWithName('test_new_array');
        assert.isDefined(method);
        const stmt = method?.getCfg()?.getStmts()[2];
        assert.isTrue(stmt instanceof ArkAssignStmt);
        assert.isTrue((stmt as ArkAssignStmt).getRightOp() instanceof ArkNewArrayExpr);
        assert.isTrue((stmt as ArkAssignStmt).getRightOp().getType() instanceof ArrayType);
        assert.isTrue((stmt as ArkAssignStmt).getLeftOp().getType() instanceof ArrayType);
    })

    it('array case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'inferSample.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getDefaultClass().getMethodWithName('testArray');
        const stmt = method?.getCfg()?.getStmts()[2];
        assert.isTrue(stmt instanceof ArkAssignStmt);
        const type = (stmt as ArkAssignStmt).getLeftOp().getType();
        assert.isTrue(type instanceof ArrayType);
        assert.isTrue((type as ArrayType).getBaseType() instanceof NumberType);
    })

    it('array Expr case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'inferSample.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getDefaultClass().getMethodWithName('arrayExpr');
        const stmts = method?.getCfg()?.getStmts();
        assert.isDefined(stmts);
        if (stmts) {
            assert.equal(stmts[1].toString(), '%0 = newarray (number)[0]');
            assert.equal(stmts[2].toString(), '%1 = newarray (string)[0]');
            assert.equal(stmts[3].toString(), '%2 = newarray (@inferType/inferSample.ts: Sample)[0]');
            assert.equal(stmts[4].toString(), '%3 = newarray (string|@inferType/inferSample.ts: Sample)[2]');
            assert.equal(stmts[5].toString(), '%4 = newarray (any)[0]');
        }
    })

    it('array Literal case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'inferSample.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getDefaultClass().getMethodWithName('arrayLiteral');
        const stmts = method?.getCfg()?.getStmts();
        assert.isDefined(stmts);
        if (stmts) {
            assert.equal(stmts[1].toString(), '%0 = newarray (number)[3]');
            assert.equal(stmts[6].toString(), '%1 = newarray (string)[2]');
            assert.equal(stmts[12].toString(), '%3 = newarray (@inferType/inferSample.ts: Sample)[1]');
            assert.equal(stmts[15].toString(), '%4 = newarray (number|string)[2]');
            assert.equal(stmts[19].toString(), '%5 = newarray (any)[0]');
            assert.equal(stmts[23].toString(), '%7 = newarray (number|string|@inferType/inferSample.ts: Sample)[3]');
        }
    })

    it('fieldRef to ArrayRef case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'inferSample.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getDefaultClass().getMethodWithName('test_new_array');
        const stmts = method?.getCfg()?.getStmts();
        assert.isDefined(stmts);
        if (stmts) {
            assert.equal(stmts[10].toString(), 'c = %2[%3]');
            assert.equal(stmts[12].toString(), 's = %4[a]');
            assert.equal(stmts[14].toString(), 'n = %5[3]');
        }
    })


    it('demo case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'demo.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getClassWithName('StaticUserB')?.getMethodWithName('f1');
        const stmt = method?.getCfg()?.getStmts()[1];
        assert.isDefined(stmt);
        assert.isTrue((stmt as ArkAssignStmt).getLeftOp().getType() instanceof NumberType);
        assert.isTrue((stmt as ArkAssignStmt).getRightOp() instanceof ArkStaticFieldRef);
    })

    it('embed namespace case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'demo.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getDefaultClass()?.getMethodWithName('testDoubleNamespace');
        const stmt = method?.getCfg()?.getStmts().at(-2);
        assert.equal(stmt?.toString(), 'staticinvoke <@inferType/demo.ts: outer.inner.TestClass.[static]request()>()');
    })

    it('field case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'Field.ts');
        const file = projectScene.getFile(fileId);
        const method = file?.getClassWithName('C2')?.getMethodWithName('f2');
        const stmt = method?.getCfg()?.getStmts()[2];
        assert.isDefined(stmt);
        assert.isTrue((stmt as ArkAssignStmt).getLeftOp().getType() instanceof ClassType);
        assert.isTrue((stmt as ArkAssignStmt).getRightOp() instanceof ArkInstanceFieldRef);
        assert.equal(file?.getClassWithName('C1')?.getFieldWithName('s')?.getType(), StringType.getInstance());
    })

    it('field type case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'inferSample.ts');
        const file = projectScene.getFile(fileId);
        const fields = file?.getClassWithName('FieldType')?.getFields();
        if (fields) {
            const arkField = fields[0];
            assert.equal(arkField.getType().toString(), '(number|string)[]');
            assert.equal(fields[1].getType(), StringType.getInstance());
        }
    })

    it('supperClass Test case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'B.ets');
        const classB = projectScene.getFile(fileId)?.getClassWithName('ClassB');
        assert.isDefined(classB?.getSuperClass());
        assert.isTrue(classB?.getFieldWithName('field1')?.getType() instanceof AliasType);
    })

    it('alias type Test case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'B.ets');
        const aliasType = projectScene.getFile(fileId)?.getDefaultClass().getDefaultArkMethod()?.getBody()?.getAliasTypeByName('TestType');
        assert.isTrue(aliasType?.getOriginalType() instanceof AliasType);
        assert.equal((aliasType?.getOriginalType() as AliasType).getOriginalType().getTypeString(), '@inferType/Target.ets: MySpace.%AC$0<@inferType/Target.ets: MySpace.ClassTarget>');
    })

    it('constructor case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'demo.ts');
        const file = projectScene.getFile(fileId);
        const returnType = file?.getClassWithName('Test')?.getMethodWithName('constructor')
            ?.getReturnType();
        assert.isTrue(returnType instanceof ClassType);
        assert.equal((returnType as ClassType).getClassSignature().toString(), '@inferType/demo.ts: Test');
    })

    it('all case', () => {
        projectScene.getMethods().forEach(m => {
            m.getCfg()?.getStmts().forEach(s => {
                const text = s.toString();
                if (text.includes('Unknown')) {
                    logger.log(text + ' warning ' + m.getSignature().toString());
                }
            })
        })
    })

    it('methodsMap refresh', () => {
        let flag = false;
        projectScene.getMethods().forEach(m => {
            if (m.getSignature().toString().includes('SCBTransitionManager.registerUnlockTransitionController(@inferType/test1.ets: SCBUnlockTransitionController')) {
                if (projectScene.getMethod(m.getSignature()) !== null) {
                    flag = true;
                }
            }
        })
        assert.isTrue(flag);
    })

    it('union array case', () => {
        let flag = false;
        projectScene.getMethods().forEach(m => {
            if (m.getSignature().toString().includes('ISceneEvent[]|ISceneEvent')) {
                if (projectScene.getMethod(m.getSignature()) !== null) {
                    flag = true;
                }
            }
        })
        assert.isTrue(flag);
    })

    it('union currType case', () => {
        let count = 0;
        projectScene.getMethods().forEach(m => {
            if (m.getSignature().toString().includes('testCurrType')) {
                m.getCfg()?.getStmts().forEach(stmt => {
                    if (stmt instanceof ArkAssignStmt) {
                        const leftOp = stmt.getLeftOp();
                        if (leftOp.getType() instanceof UnionType &&
                            (leftOp.getType() as UnionType).getCurrType().toString() === 'string') {
                            count++;
                        }
                    }
                })
            }
        })
        assert.equal(count, 3);
    })

    it('field to ArrayRef case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'Field.ts');
        const file = projectScene.getFile(fileId);
        const stmts = file?.getClassWithName('User')?.getFieldWithName('role')?.getInitializer();
        assert.isDefined(stmts);
        if (stmts) {
            assert.equal(stmts[2].toString(), '%3 = %1[%2]');
        }
    })
})

describe("function Test", () => {
    let config: SceneConfig = new SceneConfig();
    config.getSdksObj().push({ moduleName: "", name: "etsSdk", path: path.join(__dirname, "../resources/Sdk") })
    config.buildFromProjectDir(path.join(__dirname, "../resources/inferType"));
    let scene: Scene = new Scene();
    scene.buildSceneFromProjectDir(config);
    scene.inferTypes();
    it('generic case', () => {
        const fileId = new FileSignature(scene.getProjectName(), 'test2.ets');
        const file = scene.getFile(fileId);
        const actual = file?.getClassWithName('SCBSceneSessionManager')
            ?.getFieldWithName('property1')?.getSignature().getType().toString();
        assert.equal(actual, '@etsSdk/arkts/@arkts.collections.d.ets: collections.Array<number>');
    })

    it('overload case', () => {
        const fileId = new FileSignature(scene.getProjectName(), 'test2.ets');
        const file = scene.getFile(fileId);
        const actual = file?.getDefaultClass()?.getMethodWithName('demoCallBack')
            ?.getCfg()?.getStmts();
        assert.equal((actual?.[1] as ArkInvokeStmt).getInvokeExpr().getMethodSignature().toString(),
            '@etsSdk/api/@ohos.multimedia.media.d.ts: media.%dflt.createAVPlayer(AsyncCallback<@etsSdk/api/@ohos.multimedia.media.d.ts: media.AVPlayer>)');
        assert.equal((actual?.[2] as ArkAssignStmt).getInvokeExpr()?.getMethodSignature().toString(),
            '@etsSdk/api/@ohos.multimedia.media.d.ts: media.%dflt.createAVPlayer()');
    })

    it('callback case', () => {
        const fileId = new FileSignature(scene.getProjectName(), 'test2.ets');
        const file = scene.getFile(fileId);
        const actual = file?.getDefaultClass()?.getMethodWithName('%AM0$demoCallBack')
            ?.getCfg()?.getStmts().find(s => s instanceof ArkInvokeStmt)?.toString();
        assert.equal(actual, 'instanceinvoke player.<@etsSdk/api/@ohos.multimedia.media.d.ts: media.AVPlayer.on(string, Callback<drm.MediaKeySystemInfo[]>)>(\'audioInterrupt\', %AM1$%AM0$demoCallBack)');
    })

    it('promise case', () => {
        const fileId = new FileSignature(scene.getProjectName(), 'test2.ets');
        const file = scene.getFile(fileId);
        const actual2 = file?.getDefaultClass()?.getMethodWithName('%AM3$demoCallBack')
            ?.getCfg()?.getStmts().find(s => s instanceof ArkInvokeStmt)?.toString();
        assert.equal(actual2, 'instanceinvoke player.<@etsSdk/api/@ohos.multimedia.media.d.ts: media.AVPlayer.on(string, Callback<drm.MediaKeySystemInfo[]>)>(\'audioInterrupt\', %AM4$%AM3$demoCallBack)');
    })
})