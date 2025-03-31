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

import { BasicBlock } from '../../src';
import { assert, expect } from 'vitest';

export function assertBlocksEqual(blocks: Set<BasicBlock>, expectBlocks: any[]): void {
    expect(blocks.size).toEqual(expectBlocks.length);

    const blockMap = new Map<number, BasicBlock>();
    for (const block of blocks) {
        blockMap.set(block.getId(), block);
    }
    for (let i = 0; i < expectBlocks.length; i++) {
        const blockId = expectBlocks[i].id;
        const block = blockMap.get(blockId);
        if (!block) {
            assert.isDefined(block);
            return;
        }

        const stmts: string[] = [];
        for (const stmt of block.getStmts()) {
            stmts.push(stmt.toString());
        }
        expect(stmts).toEqual(expectBlocks[i].stmts);

        const preds: number[] = [];
        block.getPredecessors().forEach(predBlock => {
            preds.push(predBlock.getId());
        });
        expect(preds).toEqual(expectBlocks[i].preds);

        const succes: number[] = [];
        block.getSuccessors().forEach(succBlock => {
            succes.push(succBlock.getId());
        });
        expect(succes).toEqual(expectBlocks[i].succes);
    }
}