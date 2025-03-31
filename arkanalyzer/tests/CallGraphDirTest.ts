import { CallGraph, CallGraphBuilder, MethodSignature, Scene, SceneConfig } from "../src";
import fs from 'fs';
import path from 'path';

const DIR_PATH = 'tests/projects/ts_files';
const TARGET_PATH = 'ts_cg';

function buildCallGraph(fileName: string) {
    let config: SceneConfig = new SceneConfig();
    config.buildFromProjectFiles('test', DIR_PATH, [path.join(DIR_PATH, fileName)]);

    let scene: Scene = new Scene();
    scene.buildSceneFromFiles(config);
    scene.inferTypes();

    let entryMethods: MethodSignature[] = scene.getMethods().flatMap(method => method.getSignature());
    let cg = new CallGraph(scene);
    let cgBuilder = new CallGraphBuilder(cg, scene);
    cgBuilder.buildClassHierarchyCallGraph(entryMethods);

    dumpJson(fileName, cg);
}

function dumpJson(fileName: string, callGraph: CallGraph) {
    const jsonOutput: { [key: string]: string[] } = {};
    callGraph.getDynEdges().forEach((calledMethods, callingMethod) => {
        const callingMethodName = callingMethod.toString();
        const calledMethodNames = Array.from(calledMethods).map(method => method.toString());
        jsonOutput[callingMethodName] = calledMethodNames;
    });
    path;

    const jsonString = JSON.stringify(jsonOutput, null, 2);

    fs.writeFile(path.join(TARGET_PATH, fileName+'.json'), jsonString, 'utf-8', (err) => {
        if (err) {
            console.error(`Failed to write JSON data to ${path}:`, err);
        } else {
            console.log(`JSON data has been written to ${path}`);
        }
    });
}

function test() {
    const files = fs.readdirSync(DIR_PATH);
    files.forEach(file => {
        buildCallGraph(file);
    })
}

test();