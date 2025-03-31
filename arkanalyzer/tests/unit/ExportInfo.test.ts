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
import { ArkClass, FileSignature, Local, Scene, SceneConfig } from '../../src';
import { ArkExport, ExportInfo } from '../../src/core/model/ArkExport';
import { ArkBaseModel, ModifierType } from '../../src/core/model/ArkBaseModel';
import {
    DefaultExportClassWithDeclaring_Expect_IR,
    ExportClassWithDeclaring_Expect_IR,
    ExportClass_Expect_IR,
    ExportClassWithAs_Expect_IR,
    DefaultSingleExportClass_Expect_IR,
    DefaultExportClassInstanceWithNewIndependent_Expect_IR,
    DefaultExportClassInstanceWithNew_Expect_IR,
    DefaultExportClassInstanceWithLetIndependent_Expect_IR,
    ExportClassInstanceWithLet_Expect_IR
} from '../resources/exports/class/expectedIR';
import {
    ExportAllFromOtherFile_Expect_IR,
    ExportAllFromThisFile_Expect_IR,
    ExportAllWithAsNameFromOtherFile_Expect_IR,
    ExportAllWithAsNameFromThisFile_Expect_IR,
} from '../resources/exports/from/expectedIR';
import { DefaultExportObjectLiteral_Expect_IR } from '../resources/exports/objectLiteral/expectedIR';

function buildScene(): Scene {
    let config: SceneConfig = new SceneConfig();
    config.getSdksObj().push({ moduleName: '', name: 'etsSdk', path: path.join(__dirname, '../resources/Sdk') });
    config.getSdksObj().push({
        moduleName: '',
        name: 'lottie',
        path: path.join(__dirname, '../resources/lottieModule'),
    });
    config.buildFromProjectDir(path.join(__dirname, '../resources/exports'));
    let projectScene: Scene = new Scene();
    projectScene.buildSceneFromProjectDir(config);
    projectScene.inferTypes();
    return projectScene;
}

let projectScene = buildScene();

function compareModifiers(arkModel: ArkBaseModel | ArkExport, expectedModifiers: any): void {
    if (expectedModifiers.includes('EXPORT')) {
        assert.isTrue(arkModel.containsModifier(ModifierType.EXPORT));
    } else {
        assert.isFalse(arkModel.containsModifier(ModifierType.EXPORT));
    }
    if (expectedModifiers.includes('DEFAULT')) {
        assert.isTrue(arkModel.containsModifier(ModifierType.DEFAULT));
    } else {
        assert.isFalse(arkModel.containsModifier(ModifierType.DEFAULT));
    }
}

function compareExportInfo(exportInfo: ExportInfo | undefined, expectIR: any): void {
    assert.isDefined(exportInfo);
    assert.equal(exportInfo!.isDefault(), expectIR._default);
    assert.equal(exportInfo!.getNameBeforeAs(), expectIR.nameBeforeAs);
    assert.equal(exportInfo!.getExportClauseType(), expectIR.exportClauseType);
    if (expectIR.modifiers !== undefined) {
        compareModifiers(exportInfo as ExportInfo, expectIR.modifiers);
    }

    const arkExport = exportInfo!.getArkExport();
    assert.isDefined(arkExport);
    assert.isNotNull(arkExport);
    assert.isTrue(arkExport instanceof expectIR.arkExport.type);
    if (expectIR.arkExport.type === ArkClass) {
        assert.equal((arkExport as ArkClass).getSignature().toString(), expectIR.arkExport.classSignature);
        if (expectIR.arkExport.modifiers !== undefined) {
            compareModifiers(arkExport as ArkExport, expectIR.arkExport.modifiers);
        }
    } else if (expectIR.arkExport.type === Local) {
        if (expectIR.arkExport.local.type === ArkClass) {
            assert.equal((arkExport as Local).getType().toString(), expectIR.arkExport.local.classSignature);
        }
    }
}

describe("export Test", () => {

    it('debug case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'test.ts');
        const file = projectScene.getFile(fileId);
        assert.equal(file?.getExportInfos().length, 2);
        assert.equal(file?.getImportInfos().length, 2);
        const stmts = file?.getDefaultClass().getMethodWithName('cc')?.getCfg()?.getStmts();
        assert.isNotEmpty(stmts);
    })

    it('namespace export case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'test.ts');
        const file = projectScene.getFile(fileId);
        const stmts = file?.getDefaultClass().getMethodWithName('cc')?.getCfg()?.getStmts();
        assert.isNotEmpty(stmts);
        if (stmts) {
            assert.equal(stmts[10].toString(), 'staticinvoke <@exports/exportSample.ts: %dflt.write()>()');
            assert.equal(stmts[2].toString(), '%0 = @exports/exportSample.ts: %dflt.[static]z');
            assert.equal(stmts[9].toString(), '%2 = @exports/exportSample.ts: %dflt.[static]MyNameSpace');
        }
    })

    it('supperClass Test case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'exportSample.ts');
        assert.isDefined(projectScene.getFile(fileId)?.getClassWithName('d')?.getSuperClass());
    })

    it('import index case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'exportSample.ts');
        assert.isNotNull(projectScene.getFile(fileId)?.getImportInfoBy('Constants')?.getLazyExportInfo());
    })

    it('sdk case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'test.ts');
        assert.isDefined(projectScene.getFile(fileId)?.getImportInfoBy('hilog')?.getLazyExportInfo());
    })

    it('namespace case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'else.ts');
        const stmts = projectScene.getFile(fileId)?.getDefaultClass()
            .getMethodWithName('something')?.getCfg()?.getStmts();
        assert.isNotEmpty(stmts);
        if (stmts) {
            assert.equal(stmts[2].toString(), 'staticinvoke <@etsSdk/api/@ohos.web.webview.d.ts: webview.WebviewController.[static]setWebDebuggingAccess(boolean)>(false)');
            assert.equal(stmts[6].toString(), 'instanceinvoke controller.<@etsSdk/api/@ohos.web.webview.d.ts: webview.WebviewController.loadUrl(string|Resource, @etsSdk/api/@ohos.web.webview.d.ts: webview.WebHeader[])>(\'\')')
            assert.equal(stmts[7].toString(), 'staticinvoke <@etsSdk/api/@ohos.hilog.d.ts: hilog.%dflt.info(number, string, string, any[])>(0, \'func\', \'%{public}\', \'Ability onCreate\')')
        }
    })

    it('thirdModule case', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'Lottie_Report.ets');
        const arkExport = projectScene.getFile(fileId)?.getImportInfoBy('lottie')
            ?.getLazyExportInfo()?.getArkExport();
        assert.isTrue(arkExport instanceof Local);
    });

    it('all case', () => {
        let unknownCount = 0;
        projectScene.getMethods(true).forEach(m => {
            m.getCfg()?.getStmts().forEach(s => {
                const text = s.toString();
                if (text.includes('Unknown')) {
                    unknownCount++;
                    console.debug(text + ' warning ' + m.getSignature().toString());
                }
            })
        })
        expect(unknownCount).lessThanOrEqual(9);
    })
})

describe("function Test", () => {
    it('thirdModule index case', () => {
        let config: SceneConfig = new SceneConfig();
        config.getSdksObj().push({ moduleName: "", name: "etsSdk", path: path.join(__dirname, "../resources/Sdk") })
        config.getSdksObj().push({
            moduleName: "",
            name: "lottie",
            path: path.join(__dirname, "../resources/lottieModule")
        });
        config.buildFromProjectDir(path.join(__dirname, "../resources/exports"));
        let scene: Scene = new Scene();
        scene.buildSceneFromProjectDir(config);
        scene.inferTypes();
        const fileId = new FileSignature(scene.getProjectName(), 'Lottie_Report.ets');
        const signature = scene.getFile(fileId)?.getImportInfoBy('lottie')?.getLazyExportInfo()
            ?.getArkExport()?.toString();
        assert.equal(signature, 'Lottie')
    })
})

describe("export Class Test", () => {
    it('default export class when declaring class', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassNormal.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(DefaultExportClassWithDeclaring_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, DefaultExportClassWithDeclaring_Expect_IR);
    });

    it('export class independently from class declaration', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassNormal.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportClass_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportClass_Expect_IR);
    });

    it('export class independently from class declaration by renaming', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassNormal.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportClassWithAs_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportClassWithAs_Expect_IR);
    });

    it('export class when declaring class', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassNormal.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportClassWithDeclaring_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportClassWithDeclaring_Expect_IR);
    });

    it('single export class independently from class declaration', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/singleExportClass.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(DefaultSingleExportClass_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, DefaultSingleExportClass_Expect_IR);
    });

    it('default export class instance with new independently', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassInstanceWithNewIndependent.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(DefaultExportClassInstanceWithNewIndependent_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, DefaultExportClassInstanceWithNewIndependent_Expect_IR);
    });

    it('default export class instance with new', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassInstanceWithNew.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(DefaultExportClassInstanceWithNew_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, DefaultExportClassInstanceWithNew_Expect_IR);
    });

    it('default export class instance with independent let', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassInstanceWithLet.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(DefaultExportClassInstanceWithLetIndependent_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, DefaultExportClassInstanceWithLetIndependent_Expect_IR);
    });

    it('export class instance with let', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'class/exportClassInstanceWithLet.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportClassInstanceWithLet_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportClassInstanceWithLet_Expect_IR);
    });
})

describe("export ObjectLiteral Test", () => {
    it('default export ObjectLiteral', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'objectLiteral/exportObjectLiteral.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(DefaultExportObjectLiteral_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, DefaultExportObjectLiteral_Expect_IR);

        const anonymousClass = file?.getClassWithName('%AC$%dflt$%dflt$0');
        assert.isDefined(anonymousClass);
        assert.isNotNull(anonymousClass);
        assert.isTrue((anonymousClass as ArkClass).getFields().some(field => field.getName() === 'data'));
        assert.isTrue((anonymousClass as ArkClass).getMethods().some(m => m.getName() === 'onCreate'));
        assert.isTrue((anonymousClass as ArkClass).getMethods().some(m => m.getName() === 'onDestroy'));
    });
})

describe("export From Test", () => {
    it('export all from this file', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'from/exportAllFromThisFile.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportAllFromThisFile_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportAllFromThisFile_Expect_IR);
    });

    it('export all with as name from this file', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'from/exportAllFromThisFile.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportAllWithAsNameFromThisFile_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportAllWithAsNameFromThisFile_Expect_IR);
    });

    it('export all from other file', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'from/exportAllFromOtherFile.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportAllFromOtherFile_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportAllFromOtherFile_Expect_IR);
    });

    it('export all with as name from other file', () => {
        const fileId = new FileSignature(projectScene.getProjectName(), 'from/exportAllFromOtherFile.ts');
        const file = projectScene.getFile(fileId);
        const exportInfo = file?.getExportInfoBy(ExportAllWithAsNameFromOtherFile_Expect_IR.exportClauseName);
        compareExportInfo(exportInfo, ExportAllWithAsNameFromOtherFile_Expect_IR);
    });
})