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

import Logger, { LOG_LEVEL, LOG_MODULE_TYPE } from '../src/utils/logger';
import { SceneConfig } from '../src/Config';
import { Scene } from '../src/Scene';
import { ArkBody } from '../src/core/model/ArkBody';
import { ArkInstanceFieldRef, PrinterBuilder, Stmt } from '../src';
import { ModelUtils } from '../src/core/common/ModelUtils';
import { INSTANCE_INIT_METHOD_NAME } from '../src/core/common/Const';

const logPath = 'out/ArkAnalyzer.log';
const logger = Logger.getLogger(LOG_MODULE_TYPE.TOOL, 'ArkIRTransformerTest');
Logger.configure(logPath, LOG_LEVEL.DEBUG, LOG_LEVEL.DEBUG);

class ArkIRTransformerTest {
    public testStmtsOfSimpleProject() {
        logger.error('testStmtsOfSimpleProject start');

        const projectDir = 'tests/resources/arkIRTransformer/mainModule';
        const sceneConfig: SceneConfig = new SceneConfig();
        sceneConfig.buildFromProjectDir(projectDir);

        const scene = new Scene();
        scene.buildSceneFromProjectDir(sceneConfig);
        logger.error('\nbefore inferTypes');
        this.printScene(scene);
        scene.inferTypes();
        logger.error('\nafter inferTypes');
        this.printScene(scene);

        logger.error('testStmtsOfSimpleProject end\n');
    }

    public testStmtsOfEtsProject() {
        logger.error('testStmtsOfEtsProject start');

        // build config
        const configPath = 'tests/resources/arkIRTransformer/ArkIRTransformerTestConfig.json';
        const sceneConfig: SceneConfig = new SceneConfig();
        sceneConfig.buildFromJson(configPath);

        // build scene
        const scene = new Scene();
        scene.buildBasicInfo(sceneConfig);
        scene.buildScene4HarmonyProject();
        this.printScene(scene);
        scene.inferTypes();
        logger.error('\nafter inferTypes');
        this.printScene(scene);

        logger.error('testStmtsOfEtsProject end\n');
    }

    private printStmts(body: ArkBody): void {
        logger.error('--- threeAddressStmts ---');
        const cfg = body.getCfg();
        for (const threeAddressStmt of cfg.getStmts()) {
            logger.error(`text: '${threeAddressStmt.toString()}'`);
        }
    }

    public printOperandOriginalPositions(stmt: Stmt): void {
        const operandOriginalPositions: any[] = [];
        for (const operand of stmt.getDefAndUses()) {
            const operandOriginalPosition = stmt.getOperandOriginalPosition(operand);
            if (operandOriginalPosition) {
                operandOriginalPositions.push(
                    [operandOriginalPosition.getFirstLine(), operandOriginalPosition.getFirstCol(),
                        operandOriginalPosition.getLastLine(), operandOriginalPosition.getLastCol()]);
            } else {
                operandOriginalPositions.push(operandOriginalPosition);
            }
        }
        logger.error(`operandOriginalPositions: [${operandOriginalPositions.join('], [')}]`);
    }

    private printScene(scene: Scene): void {
        for (const arkFile of scene.getFiles()) {
            logger.error('+++++++++++++ arkFile:', arkFile.getFilePath(), ' +++++++++++++');
            for (const arkClass of ModelUtils.getAllClassesInFile(arkFile)) {
                logger.error('========= arkClass:', arkClass.getSignature().toString(), ' =======');
                for (const arkMethod of arkClass.getMethods(true)) {
                    logger.error('***** arkMethod: ', arkMethod.getName());
                    const body = arkMethod.getBody();
                    if (body) {
                        this.printStmts(body);
                        logger.error('-- locals:');
                        body.getLocals().forEach(local => {
                            logger.error(`name: ${local.getName()}, type: ${local.getType()}`);
                        });
                        logger.error('-- usedGlobals:');
                        body.getUsedGlobals()?.forEach(usedGlobalName => {
                            logger.error(`name: ${usedGlobalName}`);
                        });
                    }
                }
            }
        }
    }

    public printCfg() {
        const projectDir = 'tests/resources/arkIRTransformer/mainModule';
        const sceneConfig: SceneConfig = new SceneConfig();
        sceneConfig.buildFromProjectDir(projectDir);
        const scene = new Scene();
        scene.buildSceneFromProjectDir(sceneConfig);

        const printerBuilder = new PrinterBuilder('out');
        for (const arkFile of scene.getFiles()) {
            printerBuilder.dumpToDot(arkFile);
        }
    }

    public simpleTest() {
        logger.error('simpleTest start');
        const projectDir = 'tests/resources/arkIRTransformer/mainModule';
        const sceneConfig: SceneConfig = new SceneConfig();
        sceneConfig.buildFromProjectDir(projectDir);
        const scene = new Scene();
        scene.buildSceneFromProjectDir(sceneConfig);

        const cfg = scene.getFiles().find((file) => file.getName().endsWith(`main.ts`))
            ?.getClassWithName('Main')
            ?.getMethodWithName(INSTANCE_INIT_METHOD_NAME)?.getBody()?.getCfg();
        if (cfg) {
            const thisLocal1 = cfg.getStmts()[0].getDef();
            const thisLocal2 = (cfg.getStmts()[1].getDef() as ArkInstanceFieldRef).getBase();
            if (thisLocal1) {
                logger.log(`thisLocal1 equal to thisLocal2: ${thisLocal1 === thisLocal2}`);
            }
        } else {
            logger.log(`cfg is undefined`);
        }

        logger.error('simpleTest end');
    }
}

const arkIRTransformerTest = new ArkIRTransformerTest();
// arkIRTransformerTest.testSimpleStmt();
arkIRTransformerTest.testStmtsOfSimpleProject();
// arkIRTransformerTest.testStmtsOfEtsProject();
// arkIRTransformerTest.printCfg();
// arkIRTransformerTest.simpleTest();

