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

import { SceneConfig } from '../src/Config';
import { Scene } from '../src/Scene';
import { DEFAULT_ARK_METHOD_NAME } from '../src';

export class Test {
    public buildScene(): Scene {
        const config_path = "tests\\resources\\cfg\\CfgTestConfig.json";
        let config: SceneConfig = new SceneConfig();
        config.buildFromJson(config_path);
        return new Scene(config);
    }

    public test() {
        let scene = this.buildScene();
        scene.inferTypes();
        const classMap = scene.getClassMap();

        for (const arkFile of scene.getFiles()) {
            for (const arkClass of arkFile.getClasses()) {
                for (const arkMethod of arkClass.getMethods()) {
                    if (arkMethod.getName() == DEFAULT_ARK_METHOD_NAME) {
                        continue;
                    }
                    console.log('*** arkMethod: ', arkMethod.getName());

                    const body = arkMethod.getBody();
                    const blocks = [...body.getCfg().getBlocks()]
                    for (let i = 0; i < blocks.length; i++){

                        const block = blocks[i]
                        console.log("block"+i)
                        for (const stmt of block.getStmts()){
                            console.log("  " + stmt.toString())
                        }
                        let text = "next:"
                        for (const next of block.getSuccessors()){
                            text += blocks.indexOf(next) + ' ';
                        }
                        console.log(text);
                    }
                    
                    
                }
            }
        }
    }



    public testTypeInference(): void {
        let scene = this.buildScene();
        scene.inferTypes();
    }
}

let t = new Test();
t.test();
