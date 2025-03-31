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

import { SceneConfig } from "../src/Config";
import { Scene } from "../src/Scene";
import { MethodSignature } from "../src/core/model/ArkSignature";
import { CallGraph } from '../src/callgraph/model/CallGraph';
import { CallGraphBuilder } from '../src/callgraph/model/builder/CallGraphBuilder';
import { DEFAULT_ARK_CLASS_NAME, DEFAULT_ARK_METHOD_NAME } from '../src';

let config: SceneConfig = new SceneConfig()
// config.buildFromProjectDir('tests/resources/callgraph/test1')
config.buildFromProjectDir('tests/resources/callgraph/cha_rta_test')
// config.buildFromJson("./tests/resources/callgraph/callGraphConfigUnix.json");
// const logger = Logger.getLogger(LOG_MODULE_TYPE.ARKANALYZER, 'CG TEST');
function runScene(config: SceneConfig) {
    let projectScene: Scene = new Scene();
    projectScene.buildSceneFromProjectDir(config)
    projectScene.inferTypes()

    let entryPoints: MethodSignature[] = []
    // for (let method of projectScene.getMethods()) {
    //     entryPoints.push(method.getSignature())
    // }
    entryPoints.push(...
        projectScene.getFiles()
            .filter(arkFile => arkFile.getName() === "main.ts")
            .flatMap(arkFile => arkFile.getClasses())
            .filter(arkClass => arkClass.getName() === DEFAULT_ARK_CLASS_NAME)
            .flatMap(arkClass => arkClass.getMethods())
            .filter(arkMethod => arkMethod.getName() === "main")
            .map(arkMethod => arkMethod.getSignature())
    );

    let callGraph = new CallGraph(projectScene)
    let callGraphBuilder = new CallGraphBuilder(callGraph, projectScene)
    callGraphBuilder.buildClassHierarchyCallGraph(entryPoints, false)
    // callGraphBuilder.buildRapidTypeCallGraph(entryPoints, false)
    callGraph.dump("out/cg/cg.dot")
    // debugger;
}
runScene(config);