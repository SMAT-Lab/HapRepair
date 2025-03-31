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

import path from 'path';
import { Scene, SceneConfig } from '../../../../src';
import { assert, describe, expect, it } from 'vitest';
import { Trap } from '../../../../src/core/base/Trap';
import {
    TRAP_EXPECT_CASE1,
    TRAP_EXPECT_CASE2,
    TRAP_EXPECT_CASE3,
    TRAP_EXPECT_CASE4,
    TRAP_EXPECT_CASE5,
    TRAP_EXPECT_CASE6,
} from '../../../resources/model/body/trap/TrapExpect';
import { assertBlocksEqual } from '../../common';

const BASE_DIR = path.join(__dirname, '../../../../tests/resources/model/body');

describe('trap Test', () => {
    const scene = buildScene('trap');

    it('trap case1', async () => {
        testTraps(scene, 'TrapTest.ts', 'case1', TRAP_EXPECT_CASE1.traps);
    });

    it('trap case2', async () => {
        testTraps(scene, 'TrapTest.ts', 'case2', TRAP_EXPECT_CASE2.traps);
    });

    it('trap case3', async () => {
        testTraps(scene, 'TrapTest.ts', 'case3', TRAP_EXPECT_CASE3.traps);
    });

    it('trap case4', async () => {
        testTraps(scene, 'TrapTest.ts', 'case4', TRAP_EXPECT_CASE4.traps);
    });

    it('trap case5', async () => {
        testTraps(scene, 'TrapTest.ts', 'case5', TRAP_EXPECT_CASE5.traps);
    });

    it('trap case6', async () => {
        testTraps(scene, 'TrapTest.ts', 'case6', TRAP_EXPECT_CASE6.traps);
    });
});

function buildScene(folderName: string): Scene {
    let config: SceneConfig = new SceneConfig();
    config.buildFromProjectDir(path.join(BASE_DIR, folderName));
    let scene = new Scene();
    scene.buildSceneFromProjectDir(config);
    return scene;
}

function testTraps(scene: Scene, filePath: string, methodName: string, expectTraps: any[]): void {
    const arkFile = scene.getFiles().find((file) => file.getName().endsWith(filePath));
    const arkMethod = arkFile?.getDefaultClass().getMethods()
        .find((method) => (method.getName() === methodName));
    const traps = arkMethod?.getBody()?.getTraps();
    if (!traps) {
        assert.isDefined(traps);
        return;
    }
    assertTrapsEqual(traps, expectTraps);
}

function assertTrapsEqual(traps: Trap[], expectTraps: any[]): void {
    expect(traps.length).toEqual(expectTraps.length);
    const trapMap = new Map<string, Trap>();
    for (const trap of traps) {
        trapMap.set(generateTrapHashCode(trap), trap);
    }
    const expectTrapMap = new Map<string, any>();
    for (const expectTrap of expectTraps) {
        expectTrapMap.set(generateExpectTrapHashCode(expectTrap), expectTrap);
    }
    for (const [trapHashCode, trap] of trapMap) {
        const expectTrap = expectTrapMap.get(trapHashCode);
        if (!expectTrap) {
            assert.isDefined(expectTrap);
            return;
        }
        assertBlocksEqual(new Set(trap.getTryBlocks()), expectTrap.tryBlocks);
        assertBlocksEqual(new Set(trap.getCatchBlocks()), expectTrap.catchBlocks);
    }
}

function generateTrapHashCode(trap: Trap): string {
    const blockIds: number[] = [];
    for (const tryBlock of trap.getTryBlocks()) {
        blockIds.push(tryBlock.getId());
    }
    for (const tryBlock of trap.getCatchBlocks()) {
        blockIds.push(tryBlock.getId());
    }
    return blockIds.sort().join(',');
}

function generateExpectTrapHashCode(trap: any): string {
    const blockIds: number[] = [];
    for (const tryBlock of trap.tryBlocks) {
        blockIds.push(tryBlock.id);
    }
    for (const tryBlock of trap.catchBlocks) {
        blockIds.push(tryBlock.id);
    }
    return blockIds.sort().join(',');
}