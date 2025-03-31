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

import { Local } from '../base/Local';
import { Cfg } from '../graph/Cfg';
import { AliasType, AliasTypeDeclaration } from '../base/Type';
import { Trap } from '../base/Trap';
import { Value } from '../base/Value';

export class ArkBody {
    private locals: Map<string, Local>;
    private usedGlobals?: Map<string, Value>;
    private cfg: Cfg;
    private aliasTypeMap?: Map<string, [AliasType, AliasTypeDeclaration]>;
    private traps?: Trap[];

    constructor(locals: Set<Local>, cfg: Cfg, aliasTypeMap?: Map<string, [AliasType, AliasTypeDeclaration]>,
                traps?: Trap[]) {
        this.cfg = cfg;
        this.aliasTypeMap = aliasTypeMap;
        this.locals = new Map<string, Local>();
        locals.forEach(local => this.locals.set(local.getName(), local));
        this.traps = traps;
    }

    public getLocals(): Map<string, Local> {
        return this.locals;
    }

    public setLocals(locals: Set<Local>): void {
        if (!this.locals) {
            this.locals = new Map<string, Local>();
        }
        locals.forEach(local => this.locals.set(local.getName(), local));
    }

    public addLocal(name: string, local: Local): void {
        this.locals.set(name, local);
    }

    public getUsedGlobals(): Map<string, Value> | undefined {
        return this.usedGlobals;
    }

    public setUsedGlobals(globals: Map<string, Value>): void {
        this.usedGlobals = globals;
    }

    public getCfg(): Cfg {
        return this.cfg;
    }

    public setCfg(cfg: Cfg): void {
        this.cfg = cfg;
    }

    public getAliasTypeMap(): Map<string, [AliasType, AliasTypeDeclaration]> | undefined {
        return this.aliasTypeMap;
    }

    public getAliasTypeByName(name: string): AliasType | null {
        const aliasTypeInfo: [AliasType, AliasTypeDeclaration] | undefined = this.aliasTypeMap?.get(name);
        if (aliasTypeInfo) {
            return aliasTypeInfo[0];
        }
        return null;
    }

    public getTraps(): Trap[] | undefined {
        return this.traps;
    }
}