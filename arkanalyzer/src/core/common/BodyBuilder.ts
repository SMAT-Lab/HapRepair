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

import { ArkBody } from '../model/ArkBody';
import { ArkMethod } from '../model/ArkMethod';
import { FieldSignature, MethodSignature, MethodSubSignature } from '../model/ArkSignature';
import { CfgBuilder } from './CfgBuilder';
import * as ts from 'ohos-typescript';
import { Local } from '../base/Local';
import { MethodParameter } from '../model/builder/ArkMethodBuilder';
import { LEXICAL_ENV_NAME_PREFIX, NAME_DELIMITER, NAME_PREFIX } from './Const';
import { ArkParameterRef, ArkStaticFieldRef, ClosureFieldRef, GlobalRef } from '../base/Ref';
import { ArkAssignStmt, ArkInvokeStmt, ArkReturnStmt } from '../base/Stmt';
import { LexicalEnvType, FunctionType, ClosureType } from '../base/Type';
import { AbstractInvokeExpr, ArkPtrInvokeExpr } from '../base/Expr';

export class BodyBuilder {
    private cfgBuilder: CfgBuilder;
    private globals?: Map<string, GlobalRef>;

    constructor(methodSignature: MethodSignature, sourceAstNode: ts.Node, declaringMethod: ArkMethod, sourceFile: ts.SourceFile) {
        this.cfgBuilder = new CfgBuilder(sourceAstNode, methodSignature.getMethodSubSignature().getMethodName(), declaringMethod, sourceFile);
    }

    public build(): ArkBody | null {
        this.cfgBuilder.buildCfgBuilder();
        if (!this.cfgBuilder.isBodyEmpty()) {
            const { cfg, locals, globals, aliasTypeMap, traps } = this.cfgBuilder.buildCfgAndOriginalCfg();
            if (globals !== null) {
                this.setGlobals(globals);
            }
            cfg.buildDefUseStmt(locals);

            return new ArkBody(locals, cfg, aliasTypeMap, traps.length ? traps : undefined);
        }
        return null;
    }

    public getCfgBuilder(): CfgBuilder {
        return this.cfgBuilder;
    }

    public getGlobals(): Map<string, GlobalRef> | undefined {
        return this.globals;
    }

    public setGlobals(globals: Map<string, GlobalRef>): void {
        this.globals = globals;
    }

    public handleGlobalAndClosure(): void {
        // 首先将outerMethod中的全局变量从locals中移到usedGlobals中，避免干扰嵌套函数中闭包变量的识别
        let outerMethod = this.getCfgBuilder().getDeclaringMethod();
        let outerGlobals = outerMethod.getBodyBuilder()?.getGlobals();
        outerMethod.freeBodyBuilder();
        let outerLocals = outerMethod.getBody()?.getLocals();
        if (outerGlobals !== undefined && outerLocals !== undefined) {
            this.moveLocalToGlobal(outerLocals, outerGlobals);
            if (outerGlobals.size > 0) {
                outerMethod.getBody()?.setUsedGlobals(outerGlobals);
            }
        }

        let nestedMethods = this.findNestedMethod(outerMethod);
        if (nestedMethods === null) {
            return;
        }
        let closuresNum = 0;
        for (let nestedMethod of nestedMethods) {
            let nestedGlobals = nestedMethod.getBodyBuilder()?.getGlobals();
            nestedMethod.freeBodyBuilder();
            let nestedLocals = nestedMethod.getBody()?.getLocals();
            if (nestedLocals === undefined || nestedGlobals === undefined) {
                continue;
            }
            const nestedSignature = nestedMethod.getImplementationSignature();
            if (nestedSignature === null) {
                continue;
            }
            // 嵌套函数优先匹配闭包变量，存入词法环境中，该词法环境加入外层函数的locals中，剩余globals中变量为真正的全局变量
            let lexicalEnv = new LexicalEnvType(nestedSignature);
            if (outerLocals !== undefined) {
                this.moveGlobalToLexicalEnv(nestedGlobals, outerLocals, lexicalEnv);
                if (lexicalEnv.getClosures().length > 0) {
                    const closuresLocal = new Local(`${LEXICAL_ENV_NAME_PREFIX}${closuresNum}`, lexicalEnv);
                    outerLocals.set(closuresLocal.getName(), closuresLocal);
                    this.updateNestedMethodWithClosures(nestedMethod, closuresLocal);
                    this.updateOuterMethodWithClosures(outerMethod, nestedMethod, closuresLocal);
                    closuresNum++;
                }
            }
            this.moveLocalToGlobal(nestedLocals, nestedGlobals);
            if (nestedGlobals.size > 0) {
                nestedMethod.getBody()?.setUsedGlobals(nestedGlobals);
            }
        }
    }

    private findNestedMethod(outerMethod: ArkMethod): ArkMethod[] | null {
        let nestedMethods: ArkMethod[] = [];
        const arkClass = outerMethod.getDeclaringArkClass();
        for (let method of arkClass.getMethods()) {
            if (!method.getName().startsWith(NAME_PREFIX) || !method.getName().endsWith(outerMethod.getName())) {
                continue;
            }
            const components = method.getName().split(NAME_DELIMITER);
            // TODO: 当前仅返回第一层的嵌套函数，待支持多层嵌套
            if (components.length === 2) {
                nestedMethods.push(method);
            }
        }
        if (nestedMethods.length > 0) {
            return nestedMethods;
        }
        return null;
    }

    private moveLocalToGlobal(locals: Map<string, Local>, globals: Map<string, GlobalRef>): void {
        globals.forEach((value, key) => {
            const local = locals.get(key);
            if (local !== undefined) {
                value.addUsedStmts(local.getUsedStmts());
                locals.delete(key);
            }
        });
    }

    private moveGlobalToLexicalEnv(nestedGlobals: Map<string, GlobalRef>, outerLocals: Map<string, Local>, lexicalEnv: LexicalEnvType): void {
        for (let key of nestedGlobals.keys()) {
            const local = outerLocals.get(key);
            if (local !== undefined) {
                lexicalEnv.addClosure(local);
                nestedGlobals.delete(key);
            }
        }
    }

    private updateNestedMethodWithClosures(nestedMethod: ArkMethod, closuresLocal: Local): void {
        if (!(closuresLocal.getType() instanceof LexicalEnvType)) {
            return;
        }

        const declareSignatures = nestedMethod.getDeclareSignatures();
        declareSignatures?.forEach((signature, index) => {
            nestedMethod.setDeclareSignatureWithIndex(this.createNewSignatureWithClosures(closuresLocal, signature), index);
        });

        const implementSignature = nestedMethod.getImplementationSignature();
        if (implementSignature !== null) {
            nestedMethod.setImplementationSignature(this.createNewSignatureWithClosures(closuresLocal, implementSignature));
        }

        this.addClosureParamsAssignStmts(closuresLocal, nestedMethod);
    }

    private updateOuterMethodWithClosures(outerMethod: ArkMethod, nestedMethod: ArkMethod, closuresLocal: Local): void {
        const nestedMethodName = nestedMethod.getName();
        const nestedMethodLocal = outerMethod.getBody()?.getLocals().get(nestedMethodName);
        if (nestedMethodLocal !== undefined) {
            this.updateLocalInfoWithClosures(nestedMethodLocal, outerMethod, nestedMethod, closuresLocal);
        } else {
            const nestedMethodGlobal = outerMethod.getBody()?.getUsedGlobals()?.get(nestedMethodName);
            if (nestedMethodGlobal !== undefined && nestedMethodGlobal instanceof GlobalRef) {
                this.updateGlobalInfoWithClosures(nestedMethodGlobal, outerMethod, nestedMethod, closuresLocal);
            }
        }

        const originalMethodName = this.getOriginalNestedMethodName(nestedMethodName);
        if (originalMethodName === null) {
            return;
        }
        const originalMethodLocal = outerMethod.getBody()?.getLocals().get(originalMethodName);
        if (originalMethodLocal !== undefined) {
            this.updateLocalInfoWithClosures(originalMethodLocal, outerMethod, nestedMethod, closuresLocal);
        } else {
            const originalMethodGlobal = outerMethod.getBody()?.getUsedGlobals()?.get(originalMethodName);
            if (originalMethodGlobal !== undefined && originalMethodGlobal instanceof GlobalRef) {
                this.updateGlobalInfoWithClosures(originalMethodGlobal, outerMethod, nestedMethod, closuresLocal);
            }
        }
    }

    private getOriginalNestedMethodName(nestedMethodName: string): string | null {
        if (nestedMethodName.startsWith(NAME_PREFIX) && nestedMethodName.includes(NAME_DELIMITER)) {
            const nameComponents = nestedMethodName.slice(1).split(NAME_DELIMITER);
            if (nameComponents.length > 1) {
                return nameComponents[0];
            }
        }
        return null;
    }

    private updateGlobalInfoWithClosures(globalRef: GlobalRef, outerMethod: ArkMethod, nestedMethod: ArkMethod, closuresLocal: Local): void {
        if (globalRef.getRef() !== null) {
            return;
        }
        const methodSignature = nestedMethod.getImplementationSignature();
        if (methodSignature === null) {
            return;
        }
        const lexicalEnv = closuresLocal.getType();
        if (!(lexicalEnv instanceof LexicalEnvType)) {
            return;
        }
        const fieldSignature = new FieldSignature(
            methodSignature.getMethodSubSignature().getMethodName(),
            methodSignature.getDeclaringClassSignature(),
            new ClosureType(lexicalEnv, methodSignature));
        globalRef.setRef(new ArkStaticFieldRef((fieldSignature)));
        this.updateAbstractInvokeExprWithClosures(globalRef, outerMethod.getSignature(), nestedMethod.getSignature(), closuresLocal);
    }

    private updateLocalInfoWithClosures(local: Local, outerMethod: ArkMethod, nestedMethod: ArkMethod, closuresLocal: Local): void {
        const localType = local.getType();
        if (!(localType instanceof FunctionType)) {
            return;
        }

        const lexicalEnv = closuresLocal.getType();
        if (!(lexicalEnv instanceof LexicalEnvType)) {
            return;
        }

        // 更新local的类型为ClosureType，methodSignature为内层嵌套函数
        const nestedMethodSignature = nestedMethod.getImplementationSignature();
        if (nestedMethodSignature !== null) {
            local.setType(new ClosureType(lexicalEnv, nestedMethodSignature, localType.getRealGenericTypes()));
        } else {
            local.setType(new ClosureType(lexicalEnv, localType.getMethodSignature(), localType.getRealGenericTypes()));
        }

        this.updateAbstractInvokeExprWithClosures(local, outerMethod.getSignature(), nestedMethod.getSignature(), closuresLocal);
    }

    // 更新所有stmt中调用内层函数处的AbstractInvokeExpr中的函数签名和实参args，加入闭包参数
    // 更新所有stmt中定义的函数指针的usedStmt中的函数签名和实参args，加入闭包参数
    private updateAbstractInvokeExprWithClosures(value: Local | GlobalRef, outerMethodSignature: MethodSignature,
                                                 nestedMethodSignature: MethodSignature, closuresLocal: Local): void {
        for (const usedStmt of value.getUsedStmts()) {
            if (usedStmt instanceof ArkInvokeStmt) {
                this.updateSignatureAndArgsInArkInvokeExpr(usedStmt, nestedMethodSignature, closuresLocal);
            } else if (usedStmt instanceof ArkAssignStmt) {
                const rightOp = usedStmt.getRightOp();
                if (rightOp instanceof AbstractInvokeExpr) {
                    this.updateSignatureAndArgsInArkInvokeExpr(usedStmt, nestedMethodSignature, closuresLocal);
                }
                const leftOp = usedStmt.getLeftOp();
                if (leftOp instanceof Local) {
                    leftOp.setType(rightOp.getType());
                }
            } else if (usedStmt instanceof ArkReturnStmt) {
                outerMethodSignature.getMethodSubSignature().setReturnType(value.getType());
            }
            const defValue = usedStmt.getDef();
            if (defValue === null) {
                continue;
            }
            if ((defValue instanceof Local || defValue instanceof GlobalRef) && defValue.getType() instanceof FunctionType) {
                this.updateAbstractInvokeExprWithClosures(defValue, outerMethodSignature, nestedMethodSignature, closuresLocal);
            }
        }
    }

    private createNewSignatureWithClosures(closuresLocal: Local, oldSignature: MethodSignature): MethodSignature {
        let oldSubSignature = oldSignature.getMethodSubSignature();
        const params = oldSubSignature.getParameters();
        const closuresParam = new MethodParameter();
        closuresParam.setName(closuresLocal.getName());
        closuresParam.setType(closuresLocal.getType());
        params.unshift(closuresParam);
        let newSubSignature = new MethodSubSignature(
            oldSubSignature.getMethodName(),
            params,
            oldSubSignature.getReturnType(),
            oldSubSignature.isStatic()
        );
        return new MethodSignature(oldSignature.getDeclaringClassSignature(), newSubSignature);
    }

    private updateSignatureAndArgsInArkInvokeExpr(stmt: ArkInvokeStmt | ArkAssignStmt, methodSignature: MethodSignature, closuresLocal: Local): void {
        let expr: AbstractInvokeExpr;
        if (stmt instanceof ArkInvokeStmt) {
            expr = stmt.getInvokeExpr();
        } else {
            const rightOp = stmt.getRightOp();
            if (!(rightOp instanceof AbstractInvokeExpr)) {
                return;
            }
            expr = rightOp;
        }
        const exprMethodName = expr.getMethodSignature().getMethodSubSignature().getMethodName();
        const nestedMethodName = methodSignature.getMethodSubSignature().getMethodName();
        if (exprMethodName === nestedMethodName) {
            expr.setMethodSignature(this.createNewSignatureWithClosures(closuresLocal, methodSignature));
            expr.getArgs().unshift(closuresLocal);
            closuresLocal.addUsedStmt(stmt);
            return;
        }

        const originalMethodName = this.getOriginalNestedMethodName(nestedMethodName);
        if (originalMethodName !== null) {
            if (exprMethodName === originalMethodName || expr instanceof ArkPtrInvokeExpr) {
                expr.setMethodSignature(methodSignature);
                expr.getArgs().unshift(closuresLocal);
                closuresLocal.addUsedStmt(stmt);
            }
        }
    }

    private addClosureParamsAssignStmts(closuresParam: Local, method: ArkMethod): void {
        const lexicalEnv = closuresParam.getType();
        if (!(lexicalEnv instanceof LexicalEnvType)) {
            return;
        }
        const closures = lexicalEnv.getClosures();
        if (closures.length === 0) {
            return;
        }
        const oldParamRefs = method.getParameterRefs();
        let body = method.getBody();
        if (body === undefined) {
            return;
        }
        let stmts = Array.from(body.getCfg().getBlocks())[0].getStmts();
        let index = 0;
        const parameterRef = new ArkParameterRef(index, lexicalEnv);
        const closuresLocal = new Local(closuresParam.getName(), lexicalEnv);
        body.addLocal(closuresLocal.getName(), closuresLocal);
        let assignStmt = new ArkAssignStmt(closuresLocal, parameterRef);
        stmts.splice(index, 0, assignStmt);
        closuresLocal.setDeclaringStmt(assignStmt);

        oldParamRefs?.forEach((paramRef) => {
            index++;
            paramRef.setIndex(index);
        });

        for (let closure of closures) {
            const local = method.getBody()?.getLocals().get(closure.getName());
            if (local === undefined) {
                continue;
            }
            index++;
            const closureFieldRef = new ClosureFieldRef(closuresParam, closure.getName(), closure.getType());
            let assignStmt = new ArkAssignStmt(local, closureFieldRef);
            stmts.splice(index, 0, assignStmt);
            local.setDeclaringStmt(assignStmt);
            closuresLocal.addUsedStmt(assignStmt);
        }
    }
}