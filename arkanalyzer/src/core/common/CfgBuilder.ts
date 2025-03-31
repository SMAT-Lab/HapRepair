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

import * as ts from 'ohos-typescript';
import { Local } from '../base/Local';
import {
    ArkAssignStmt,
    ArkIfStmt,
    ArkInvokeStmt,
    ArkReturnStmt,
    ArkReturnVoidStmt,
    ArkThrowStmt,
    Stmt,
} from '../base/Stmt';
import { BasicBlock } from '../graph/BasicBlock';
import { Cfg } from '../graph/Cfg';
import { ArkClass } from '../model/ArkClass';
import { ArkMethod } from '../model/ArkMethod';
import { ArkIRTransformer, DummyStmt, ValueAndStmts } from './ArkIRTransformer';
import { ModelUtils } from './ModelUtils';
import { AbstractInvokeExpr } from '../base/Expr';
import { Builtin } from './Builtin';
import { IRUtils } from './IRUtils';
import { AliasType, AliasTypeDeclaration, UnknownType } from '../base/Type';
import { Trap } from '../base/Trap';
import Logger, { LOG_MODULE_TYPE } from '../../utils/logger';
import { ArkCaughtExceptionRef, GlobalRef } from '../base/Ref';
import { FullPosition } from '../base/Position';

const logger = Logger.getLogger(LOG_MODULE_TYPE.ARKANALYZER, 'CfgBuilder');

class StatementBuilder {
    type: string;
    //节点对应源代码    
    code: string;
    next: StatementBuilder | null;
    lasts: Set<StatementBuilder>;
    walked: boolean;
    index: number;
    // TODO:以下两个属性需要获取    
    line: number;//行号//ast节点存了一个start值为这段代码的起始地址，可以从start开始往回查原文有几个换行符确定行号    
    column: number; // 列  
    astNode: ts.Node | null;//ast节点对象
    scopeID: number;
    addressCode3: string[] = [];
    block: Block | null;
    ifExitPass: boolean;
    passTmies: number = 0;
    numOfIdentifier: number = 0;
    isDoWhile: boolean = false;

    constructor(type: string, code: string, astNode: ts.Node | null, scopeID: number) {
        this.type = type;
        this.code = code;
        this.next = null;
        this.lasts = new Set();
        this.walked = false;
        this.index = 0;
        this.line = -1;
        this.column = -1;
        this.astNode = astNode;
        this.scopeID = scopeID;
        this.block = null;
        this.ifExitPass = false;
    }
}

class ConditionStatementBuilder extends StatementBuilder {
    nextT: StatementBuilder | null;
    nextF: StatementBuilder | null;
    loopBlock: Block | null;
    condition: string;
    doStatement: StatementBuilder | null = null;

    constructor(type: string, code: string, astNode: ts.Node, scopeID: number) {
        super(type, code, astNode, scopeID);
        this.nextT = null;
        this.nextF = null;
        this.loopBlock = null;
        this.condition = '';
    }
}

class SwitchStatementBuilder extends StatementBuilder {
    nexts: StatementBuilder[];
    cases: Case[] = [];
    default: StatementBuilder | null = null;
    afterSwitch: StatementBuilder | null = null;

    constructor(type: string, code: string, astNode: ts.Node, scopeID: number) {
        super(type, code, astNode, scopeID);
        this.nexts = [];
    }
}

class TryStatementBuilder extends StatementBuilder {
    tryFirst: StatementBuilder | null = null;
    tryExit: StatementBuilder | null = null;
    catchStatement: StatementBuilder | null = null;
    catchError: string = '';
    finallyStatement: StatementBuilder | null = null;
    afterFinal: StatementBuilder | null = null;

    constructor(type: string, code: string, astNode: ts.Node, scopeID: number) {
        super(type, code, astNode, scopeID);
    }
}

class Case {
    value: string;
    stmt: StatementBuilder;
    valueNode!: ts.Node;

    constructor(value: string, stmt: StatementBuilder) {
        this.value = value;
        this.stmt = stmt;
    }
}

class DefUseChain {
    def: StatementBuilder;
    use: StatementBuilder;

    constructor(def: StatementBuilder, use: StatementBuilder) {
        this.def = def;
        this.use = use;
    }
}

class Variable {
    name: string;
    lastDef: StatementBuilder;
    defUse: DefUseChain[];
    properties: Variable[] = [];
    propOf: Variable | null = null;

    constructor(name: string, lastDef: StatementBuilder) {
        this.name = name;
        this.lastDef = lastDef;
        this.defUse = [];
    }
}

class Scope {
    id: number;

    constructor(id: number) {
        this.id = id;
    }
}

class Block {
    id: number;
    stmts: StatementBuilder[];
    nexts: Block[] = [];
    lasts: Block[] = [];
    walked: boolean = false;

    constructor(id: number, stmts: StatementBuilder[]) {
        this.id = id;
        this.stmts = stmts;
    }
}

class Catch {
    errorName: string;
    from: number;
    to: number;
    withLabel: number;

    constructor(errorName: string, from: number, to: number, withLabel: number) {
        this.errorName = errorName;
        this.from = from;
        this.to = to;
        this.withLabel = withLabel;
    }
}

class textError extends Error {
    constructor(message: string) {
        // 调用父类的构造函数，并传入错误消息
        super(message);

        // 设置错误类型的名称
        this.name = 'textError';
    }
}

export class CfgBuilder {
    name: string;
    astRoot: ts.Node;
    entry: StatementBuilder;
    exit: StatementBuilder;
    loopStack: ConditionStatementBuilder[];
    switchExitStack: StatementBuilder[];
    functions: CfgBuilder[];
    breakin: string;
    statementArray: StatementBuilder[];
    dotEdges: number[][];
    scopes: Scope[];
    tempVariableNum: number;
    current3ACstm: StatementBuilder;
    blocks: Block[];
    currentDeclarationKeyword: string;
    variables: Variable[];
    declaringClass: ArkClass;
    importFromPath: string[];
    catches: Catch[];
    exits: StatementBuilder[] = [];
    emptyBody: boolean = false;
    arrowFunctionWithoutBlock: boolean = false;

    private sourceFile: ts.SourceFile;
    private declaringMethod: ArkMethod;

    constructor(ast: ts.Node, name: string, declaringMethod: ArkMethod, sourceFile: ts.SourceFile) {
        this.name = name;
        this.astRoot = ast;
        this.declaringMethod = declaringMethod;
        this.declaringClass = declaringMethod.getDeclaringArkClass();
        this.entry = new StatementBuilder('entry', '', ast, 0);
        this.loopStack = [];
        this.switchExitStack = [];
        this.functions = [];
        this.breakin = '';
        this.statementArray = [];
        this.dotEdges = [];
        this.exit = new StatementBuilder('exit', 'return;', null, 0);
        this.scopes = [];
        this.tempVariableNum = 0;
        this.current3ACstm = this.entry;
        this.blocks = [];
        this.currentDeclarationKeyword = '';
        this.variables = [];
        this.importFromPath = [];
        this.catches = [];
        this.sourceFile = sourceFile;
        this.arrowFunctionWithoutBlock = true;
    }

    public getDeclaringMethod(): ArkMethod {
        return this.declaringMethod;
    }

    judgeLastType(s: StatementBuilder, lastStatement: StatementBuilder): void {
        if (lastStatement.type === 'ifStatement') {
            let lastIf = lastStatement as ConditionStatementBuilder;
            if (lastIf.nextT == null) {
                lastIf.nextT = s;
                s.lasts.add(lastIf);
            } else {
                lastIf.nextF = s;
                s.lasts.add(lastIf);
            }
        } else if (lastStatement.type === 'loopStatement') {
            let lastLoop = lastStatement as ConditionStatementBuilder;
            lastLoop.nextT = s;
            s.lasts.add(lastLoop);
        } else if (lastStatement.type === 'catchOrNot') {
            let lastLoop = lastStatement as ConditionStatementBuilder;
            lastLoop.nextT = s;
            s.lasts.add(lastLoop);
        } else {
            lastStatement.next = s;
            s.lasts.add(lastStatement);
        }

    }

    ASTNodeBreakStatement(c: ts.Node, lastStatement: StatementBuilder): void {
        let p: ts.Node | null = c;
        while (p && p !== this.astRoot) {
            if (ts.isWhileStatement(p) || ts.isDoStatement(p) || ts.isForStatement(p) || ts.isForInStatement(p) || ts.isForOfStatement(p)) {
                const lastLoopNextF = this.loopStack[this.loopStack.length - 1].nextF!;
                this.judgeLastType(lastLoopNextF, lastStatement);
                lastLoopNextF.lasts.add(lastStatement);
                return;
            }
            if (ts.isCaseClause(p) || ts.isDefaultClause(p)) {
                const lastSwitchExit = this.switchExitStack[this.switchExitStack.length - 1];
                this.judgeLastType(lastSwitchExit, lastStatement);
                lastSwitchExit.lasts.add(lastStatement);
                return;
            }
            p = p.parent;
        }
    }

    ASTNodeIfStatement(c: ts.IfStatement, lastStatement: StatementBuilder, scopeID: number): StatementBuilder {
        let ifstm: ConditionStatementBuilder = new ConditionStatementBuilder('ifStatement', '', c, scopeID);
        this.judgeLastType(ifstm, lastStatement);
        let ifexit: StatementBuilder = new StatementBuilder('ifExit', '', c, scopeID);
        this.exits.push(ifexit);
        ifstm.condition = c.expression.getText(this.sourceFile);
        ifstm.code = 'if (' + ifstm.condition + ')';
        if (ts.isBlock(c.thenStatement)) {
            this.walkAST(ifstm, ifexit, [...c.thenStatement.statements]);
        } else {
            this.walkAST(ifstm, ifexit, [c.thenStatement]);
        }
        if (c.elseStatement) {
            if (ts.isBlock(c.elseStatement)) {
                this.walkAST(ifstm, ifexit, [...c.elseStatement.statements]);
            } else {
                this.walkAST(ifstm, ifexit, [c.elseStatement]);
            }
        }
        if (!ifstm.nextT) {
            ifstm.nextT = ifexit;
            ifexit.lasts.add(ifstm);
        }
        if (!ifstm.nextF) {
            ifstm.nextF = ifexit;
            ifexit.lasts.add(ifstm);
        }
        return ifexit;
    }

    ASTNodeWhileStatement(c: ts.WhileStatement, lastStatement: StatementBuilder, scopeID: number): StatementBuilder {
        this.breakin = 'loop';
        let loopstm = new ConditionStatementBuilder('loopStatement', '', c, scopeID);
        this.loopStack.push(loopstm);
        this.judgeLastType(loopstm, lastStatement);
        let loopExit = new StatementBuilder('loopExit', '', c, scopeID);
        this.exits.push(loopExit);
        loopstm.nextF = loopExit;
        loopExit.lasts.add(loopstm);
        loopstm.condition = c.expression.getText(this.sourceFile);
        loopstm.code = 'while (' + loopstm.condition + ')';
        if (ts.isBlock(c.statement)) {
            this.walkAST(loopstm, loopstm, [...c.statement.statements]);
        } else {
            this.walkAST(loopstm, loopstm, [c.statement]);
        }
        if (!loopstm.nextF) {
            loopstm.nextF = loopExit;
            loopExit.lasts.add(loopstm);
        }
        if (!loopstm.nextT) {
            loopstm.nextT = loopExit;
            loopExit.lasts.add(loopstm);
        }
        this.loopStack.pop();
        return loopExit;
    }

    ASTNodeForStatement(c: ts.ForInOrOfStatement | ts.ForStatement, lastStatement: StatementBuilder, scopeID: number): StatementBuilder {
        this.breakin = 'loop';
        let loopstm = new ConditionStatementBuilder('loopStatement', '', c, scopeID);
        this.loopStack.push(loopstm);
        this.judgeLastType(loopstm, lastStatement);
        let loopExit = new StatementBuilder('loopExit', '', c, scopeID);
        this.exits.push(loopExit);
        loopstm.nextF = loopExit;
        loopExit.lasts.add(loopstm);
        loopstm.code = 'for (';
        if (ts.isForStatement(c)) {
            loopstm.code += c.initializer?.getText(this.sourceFile) + '; ' + c.condition?.getText(this.sourceFile) + '; ' + c.incrementor?.getText(this.sourceFile);
        } else if (ts.isForOfStatement(c)) {
            loopstm.code += c.initializer?.getText(this.sourceFile) + ' of ' + c.expression.getText(this.sourceFile);
        } else {
            loopstm.code += c.initializer?.getText(this.sourceFile) + ' in ' + c.expression.getText(this.sourceFile);
        }
        loopstm.code += ')';
        if (ts.isBlock(c.statement)) {
            this.walkAST(loopstm, loopstm, [...c.statement.statements]);
        } else {
            this.walkAST(loopstm, loopstm, [c.statement]);
        }
        if (!loopstm.nextF) {
            loopstm.nextF = loopExit;
            loopExit.lasts.add(loopstm);
        }
        if (!loopstm.nextT) {
            loopstm.nextT = loopExit;
            loopExit.lasts.add(loopstm);
        }
        this.loopStack.pop();
        return loopExit;
    }

    ASTNodeDoStatement(c: ts.DoStatement, lastStatement: StatementBuilder, scopeID: number): StatementBuilder {
        this.breakin = 'loop';
        let loopstm = new ConditionStatementBuilder('loopStatement', '', c, scopeID);
        this.loopStack.push(loopstm);
        let loopExit = new StatementBuilder('loopExit', '', c, scopeID);
        this.exits.push(loopExit);
        loopstm.nextF = loopExit;
        loopExit.lasts.add(loopstm);
        loopstm.condition = c.expression.getText(this.sourceFile);
        loopstm.code = 'while (' + loopstm.condition + ')';
        loopstm.isDoWhile = true;
        if (ts.isBlock(c.statement)) {
            this.walkAST(lastStatement, loopstm, [...c.statement.statements]);
        } else {
            this.walkAST(lastStatement, loopstm, [c.statement]);
        }
        let lastType = lastStatement.type;
        if (lastType === 'ifStatement' || lastType === 'loopStatement') {
            let lastCondition = lastStatement as ConditionStatementBuilder;
            loopstm.nextT = lastCondition.nextT;
            lastCondition.nextT?.lasts.add(loopstm);
        } else {
            loopstm.nextT = lastStatement.next;
            lastStatement.next?.lasts.add(loopstm);
        }
        if (loopstm.nextT && loopstm.nextT !== loopstm) {
            loopstm.nextT.isDoWhile = true;
            loopstm.doStatement = loopstm.nextT;
        }
        this.loopStack.pop();
        return loopExit;
    }

    ASTNodeSwitchStatement(c: ts.SwitchStatement, lastStatement: StatementBuilder, scopeID: number): StatementBuilder {
        this.breakin = 'switch';
        let switchstm = new SwitchStatementBuilder('switchStatement', '', c, scopeID);
        this.judgeLastType(switchstm, lastStatement);
        let switchExit = new StatementBuilder('switchExit', '', null, scopeID);
        this.exits.push(switchExit);
        this.switchExitStack.push(switchExit);
        switchExit.lasts.add(switchstm);
        switchstm.code = 'switch (' + c.expression + ')';
        let lastCaseExit: StatementBuilder | null = null;
        for (let i = 0; i < c.caseBlock.clauses.length; i++) {
            const clause = c.caseBlock.clauses[i];
            let casestm: StatementBuilder;
            if (ts.isCaseClause(clause)) {
                casestm = new StatementBuilder('statement', 'case ' + clause.expression.getText(this.sourceFile) + ':', clause, scopeID);
            } else {
                casestm = new StatementBuilder('statement', 'default:', clause, scopeID);
            }
            switchstm.nexts.push(casestm);
            casestm.lasts.add(switchstm);
            let caseExit = new StatementBuilder('caseExit', '', null, scopeID);
            this.exits.push(caseExit);
            this.walkAST(casestm, caseExit, [...clause.statements]);
            if (ts.isCaseClause(clause)) {
                const cas = new Case(casestm.code, casestm.next!);
                switchstm.cases.push(cas);
            } else {
                switchstm.default = casestm.next;
            }
            switchstm.nexts[switchstm.nexts.length - 1] = casestm.next!;
            for (const stmt of [...casestm.lasts]) {
                casestm.next!.lasts.add(stmt);
            }
            casestm.next!.lasts.delete(casestm);

            if (lastCaseExit) {
                lastCaseExit.next = casestm.next;
                casestm.next?.lasts.add(lastCaseExit);
            }
            lastCaseExit = caseExit;
            if (i === c.caseBlock.clauses.length - 1) {
                caseExit.next = switchExit;
                switchExit.lasts.add(caseExit);
            }
        }
        this.switchExitStack.pop();
        return switchExit;
    }

    ASTNodeTryStatement(c: ts.TryStatement, lastStatement: StatementBuilder, scopeID: number): StatementBuilder {
        let trystm = new TryStatementBuilder('tryStatement', 'try', c, scopeID);
        this.judgeLastType(trystm, lastStatement);
        let tryExit = new StatementBuilder('tryExit', '', c, scopeID);
        this.exits.push(tryExit);
        trystm.tryExit = tryExit;
        this.walkAST(trystm, tryExit, [...c.tryBlock.statements]);
        trystm.tryFirst = trystm.next;
        trystm.next?.lasts.add(trystm);
        if (c.catchClause) {
            let text = 'catch';
            if (c.catchClause.variableDeclaration) {
                text += '(' + c.catchClause.variableDeclaration.getText(this.sourceFile) + ')';
            }
            let catchOrNot = new ConditionStatementBuilder('catchOrNot', text, c, scopeID);
            let catchExit = new StatementBuilder('catch exit', '', c, scopeID);
            catchOrNot.nextF = catchExit;
            catchExit.lasts.add(catchOrNot);
            this.walkAST(catchOrNot, catchExit, [...c.catchClause.block.statements]);
            if (!catchOrNot.nextT) {
                catchOrNot.nextT = catchExit;
                catchExit.lasts.add(catchOrNot);
            }
            const catchStatement = new StatementBuilder('statement', catchOrNot.code, c.catchClause, catchOrNot.nextT.scopeID);
            catchStatement.next = catchOrNot.nextT;
            trystm.catchStatement = catchStatement;
            catchStatement.lasts.add(trystm);
            if (c.catchClause.variableDeclaration) {
                trystm.catchError = c.catchClause.variableDeclaration.getText(this.sourceFile);
            } else {
                trystm.catchError = 'Error';
            }

        }
        let final = new StatementBuilder('statement', 'finally', c, scopeID);
        let finalExit = new StatementBuilder('finallyExit', '', c, scopeID);
        this.exits.push(finalExit);
        if (c.finallyBlock && c.finallyBlock.statements.length > 0) {
            this.walkAST(final, finalExit, [...c.finallyBlock.statements]);
        } else {
            let dummyFinally = new StatementBuilder('statement', 'dummyFinally', c, (new Scope(this.scopes.length)).id);
            final.next = dummyFinally;
            dummyFinally.lasts.add(final);
            dummyFinally.next = finalExit;
            finalExit.lasts.add(dummyFinally);
        }
        trystm.finallyStatement = final.next;
        tryExit.next = final.next;
        final.next?.lasts.add(tryExit);

        trystm.next = finalExit;
        finalExit.lasts.add(trystm);
        return finalExit;
    }

    walkAST(lastStatement: StatementBuilder, nextStatement: StatementBuilder, nodes: ts.Node[]) {
        let scope = new Scope(this.scopes.length);
        this.scopes.push(scope);
        for (let i = 0; i < nodes.length; i++) {
            let c = nodes[i];
            if (ts.isVariableStatement(c) || ts.isExpressionStatement(c) || ts.isThrowStatement(c) || ts.isTypeAliasDeclaration(c)) {
                let s = new StatementBuilder('statement', c.getText(this.sourceFile), c, scope.id);
                this.judgeLastType(s, lastStatement);
                lastStatement = s;
            } else if (!this.declaringMethod.isDefaultArkMethod() && ts.isFunctionDeclaration(c)) {
                let s = new StatementBuilder('functionDeclarationStatement', c.getText(this.sourceFile), c, scope.id);
                this.judgeLastType(s, lastStatement);
                lastStatement = s;
            } else if (!this.declaringMethod.isDefaultArkMethod() && ts.isClassDeclaration(c)) {
                let s = new StatementBuilder('classDeclarationStatement', c.getText(this.sourceFile), c, scope.id);
                this.judgeLastType(s, lastStatement);
                lastStatement = s;
            } else if (ts.isReturnStatement(c)) {
                let s = new StatementBuilder('returnStatement', c.getText(this.sourceFile), c, scope.id);
                this.judgeLastType(s, lastStatement);
                lastStatement = s;
                break;
            } else if (ts.isBreakStatement(c)) {
                this.ASTNodeBreakStatement(c, lastStatement);
                return;
            } else if (ts.isContinueStatement(c)) {
                const lastLoop = this.loopStack[this.loopStack.length - 1];
                this.judgeLastType(lastLoop, lastStatement);
                lastLoop.lasts.add(lastStatement);
                return;
            } else if (ts.isIfStatement(c)) {
                lastStatement = this.ASTNodeIfStatement(c, lastStatement, scope.id);
            } else if (ts.isWhileStatement(c)) {
                lastStatement = this.ASTNodeWhileStatement(c, lastStatement, scope.id);
            }
            if (ts.isForStatement(c) || ts.isForInStatement(c) || ts.isForOfStatement(c)) {
                lastStatement = this.ASTNodeForStatement(c, lastStatement, scope.id);
            } else if (ts.isDoStatement(c)) {
                lastStatement = this.ASTNodeDoStatement(c, lastStatement, scope.id);
            } else if (ts.isSwitchStatement(c)) {
                lastStatement = this.ASTNodeSwitchStatement(c, lastStatement, scope.id);
            } else if (ts.isBlock(c)) {
                let blockExit = new StatementBuilder('blockExit', '', c, scope.id);
                this.exits.push(blockExit);
                this.walkAST(lastStatement, blockExit, c.getChildren(this.sourceFile)[1].getChildren(this.sourceFile));
                lastStatement = blockExit;
            } else if (ts.isTryStatement(c)) {
                lastStatement = this.ASTNodeTryStatement(c, lastStatement, scope.id);
            } else if (ts.isExportAssignment(c)) {
                if (ts.isNewExpression(c.expression) || ts.isObjectLiteralExpression(c.expression)) {
                    let s = new StatementBuilder('statement', c.getText(this.sourceFile), c, scope.id);
                    this.judgeLastType(s, lastStatement);
                    lastStatement = s;
                }
            }
        }
        if (lastStatement.type !== 'breakStatement' && lastStatement.type !== 'continueStatement' && lastStatement.type !== 'returnStatement') {
            lastStatement.next = nextStatement;
            nextStatement.lasts.add(lastStatement);
        }
    }

    addReturnInEmptyMethod() {
        if (this.entry.next === this.exit) {
            const ret = new StatementBuilder('returnStatement', 'return;', null, this.entry.scopeID);
            this.entry.next = ret;
            ret.lasts.add(this.entry);
            ret.next = this.exit;
            this.exit.lasts = new Set([ret]);
        }
    }

    deleteExitAfterCondition(last: ConditionStatementBuilder, exit: StatementBuilder): void {
        if (last.nextT === exit) {
            last.nextT = exit.next;
            const lasts = exit.next!.lasts;
            lasts.delete(exit);
            lasts.add(last);
        } else if (last.nextF === exit) {
            last.nextF = exit.next;
            const lasts = exit.next!.lasts;
            lasts.delete(exit);
            lasts.add(last);
        }
    }

    deleteExitAfterSwitch(last: SwitchStatementBuilder, exit: StatementBuilder): void {
        if (exit.type === 'switchExit') {
            last.afterSwitch = exit.next;
        }
        exit.next!.lasts.delete(exit);
        last.nexts = last.nexts.filter(item => item !== exit);
        if (last.nexts.length === 0) {
            last.next = exit.next;
            exit.next?.lasts.add(last);
        }
    }

    deleteExit(): void {
        for (const exit of this.exits) {
            const lasts = [...exit.lasts];
            for (const last of lasts) {
                if (last instanceof ConditionStatementBuilder) {
                    this.deleteExitAfterCondition(last, exit);
                } else if (last instanceof SwitchStatementBuilder) {
                    this.deleteExitAfterSwitch(last, exit);
                } else if (last instanceof TryStatementBuilder && exit.type === 'finallyExit') {
                    last.afterFinal = exit.next;
                    last.next = last.tryFirst;
                    exit.lasts.delete(last);
                } else {
                    last.next = exit.next;
                    const lasts = exit.next!.lasts;
                    lasts.delete(exit);
                    lasts.add(last);
                }
            }
        }
        // 部分语句例如return后面的exit语句的next无法在上面清除
        for (const exit of this.exits) {
            if (exit.next && exit.next.lasts.has(exit)) {
                exit.next.lasts.delete(exit);
            }
        }
    }

    addStmt2BlockStmtQueueInSpecialCase(stmt: StatementBuilder, stmtQueue: StatementBuilder[]): StatementBuilder | null {
        if (stmt.next) {
            if ((stmt.type === 'continueStatement' || stmt.next.type === 'loopStatement') && stmt.next.block || stmt.next.type.includes('exit')) {
                return null;
            }
            stmt.next.passTmies++;
            if (stmt.next.passTmies === stmt.next.lasts.size || stmt.next.type === 'loopStatement' || stmt.next.isDoWhile) {
                if (stmt.next.scopeID !== stmt.scopeID && !(stmt.next instanceof ConditionStatementBuilder && stmt.next.doStatement) &&
                    !(ts.isCaseClause(stmt.astNode!) || ts.isDefaultClause(stmt.astNode!))) {
                    stmtQueue.push(stmt.next);
                    return null;
                }
                return stmt.next;
            }
        }
        return null;
    }

    addStmt2BlockStmtQueue(stmt: StatementBuilder, stmtQueue: StatementBuilder[]): StatementBuilder | null {
        if (stmt instanceof ConditionStatementBuilder) {
            stmtQueue.push(stmt.nextF!);
            stmtQueue.push(stmt.nextT!);
        } else if (stmt instanceof SwitchStatementBuilder) {
            if (stmt.nexts.length === 0) {
                stmtQueue.push(stmt.afterSwitch!);
            }
            for (let i = stmt.nexts.length - 1; i >= 0; i--) {
                stmtQueue.push(stmt.nexts[i]);
            }
        } else if (stmt instanceof TryStatementBuilder) {
            if (stmt.finallyStatement) {
                stmtQueue.push(stmt.finallyStatement);
            }
            if (stmt.catchStatement) {
                stmtQueue.push(stmt.catchStatement);
            }
            if (stmt.tryFirst) {
                stmtQueue.push(stmt.tryFirst);
            }
        } else if (stmt.next) {
            return this.addStmt2BlockStmtQueueInSpecialCase(stmt, stmtQueue);
        }
        return null;
    }

    buildBlocks(): void {
        const stmtQueue = [this.entry];
        const handledStmts: Set<StatementBuilder> = new Set();
        while (stmtQueue.length > 0) {
            let stmt = stmtQueue.pop()!;
            if (stmt.type.includes('exit')) {
                continue;
            }
            if (handledStmts.has(stmt)) {
                continue;
            }
            const block = new Block(this.blocks.length, []);
            this.blocks.push(block);
            while (stmt && !handledStmts.has(stmt)) {
                if (stmt.type === 'loopStatement' && block.stmts.length > 0 && !stmt.isDoWhile) {
                    stmtQueue.push(stmt);
                    break;
                }
                if (stmt.type.includes('Exit')) {
                    break;
                }
                block.stmts.push(stmt);
                stmt.block = block;
                handledStmts.add(stmt);
                const addRet = this.addStmt2BlockStmtQueue(stmt, stmtQueue);
                if (addRet instanceof StatementBuilder) {
                    stmt = addRet;
                } else {
                    break;
                }
            }
        }
    }

    buildConditionNextBlocks(originStatement: ConditionStatementBuilder, block: Block, isLastStatement: boolean): void {
        let nextT = originStatement.nextT?.block;
        if (nextT && (isLastStatement || nextT !== block) && !originStatement.nextT?.type.includes(' exit')) {
            block.nexts.push(nextT);
            nextT.lasts.push(block);
        }
        let nextF = originStatement.nextF?.block;
        if (nextF && (isLastStatement || nextF !== block) && !originStatement.nextF?.type.includes(' exit')) {
            block.nexts.push(nextF);
            nextF.lasts.push(block);
        }
    }

    buildSwitchNextBlocks(originStatement: SwitchStatementBuilder, block: Block, isLastStatement: boolean): void {
        if (originStatement.nexts.length === 0) {
            const nextBlock = originStatement.afterSwitch!.block;
            if (nextBlock && (isLastStatement || nextBlock !== block)) {
                block.nexts.push(nextBlock);
                nextBlock.lasts.push(block);
            }
        }
        for (const next of originStatement.nexts) {
            const nextBlock = next.block;
            if (nextBlock && (isLastStatement || nextBlock !== block)) {
                block.nexts.push(nextBlock);
                nextBlock.lasts.push(block);
            }
        }
    }

    buildNormalNextBlocks(originStatement: StatementBuilder, block: Block, isLastStatement: boolean): void {
        let next = originStatement.next?.block;
        if (next && (isLastStatement || next !== block) && !originStatement.next?.type.includes(' exit')) {
            block.nexts.push(next);
            next.lasts.push(block);
        }
    }

    buildBlocksNextLast(): void {
        for (let block of this.blocks) {
            for (let originStatement of block.stmts) {
                let isLastStatement = (block.stmts.indexOf(originStatement) === block.stmts.length - 1);
                if (originStatement instanceof ConditionStatementBuilder) {
                    this.buildConditionNextBlocks(originStatement, block, isLastStatement);
                } else if (originStatement instanceof SwitchStatementBuilder) {
                    this.buildSwitchNextBlocks(originStatement, block, isLastStatement);
                } else {
                    this.buildNormalNextBlocks(originStatement, block, isLastStatement);
                }
            }
        }
    }

    addReturnBlock(returnStatement: StatementBuilder, notReturnStmts: StatementBuilder[]) {
        let returnBlock = new Block(this.blocks.length, [returnStatement]);
        returnStatement.block = returnBlock;
        this.blocks.push(returnBlock);
        for (const notReturnStmt of notReturnStmts) {
            if (notReturnStmt instanceof ConditionStatementBuilder) {
                if (this.exit === notReturnStmt.nextT) {
                    notReturnStmt.nextT = returnStatement;
                    notReturnStmt.block?.nexts.splice(0, 0, returnBlock);
                } else if (this.exit === notReturnStmt.nextF) {
                    notReturnStmt.nextF = returnStatement;
                    notReturnStmt.block?.nexts.push(returnBlock);
                }
            } else {
                notReturnStmt.next = returnStatement;
                notReturnStmt.block?.nexts.push(returnBlock);
            }
            returnStatement.lasts.add(notReturnStmt);
            returnStatement.next = this.exit;
            const lasts = [...this.exit.lasts];
            lasts[lasts.indexOf(notReturnStmt)] = returnStatement;
            this.exit.lasts = new Set(lasts);
            returnBlock.lasts.push(notReturnStmt.block!);
        }
        this.exit.block = returnBlock;
    }

    addReturnStmt(): void {
        let notReturnStmts: StatementBuilder[] = [];
        for (let stmt of [...this.exit.lasts]) {
            if (stmt.type !== 'returnStatement') {
                notReturnStmts.push(stmt);
            }
        }
        if (notReturnStmts.length < 1) {
            return;
        }
        const returnStatement = new StatementBuilder('returnStatement', 'return;', null, this.exit.scopeID);
        let TryOrSwitchExit = false;
        if (notReturnStmts.length === 1 && notReturnStmts[0].block) {
            let p: ts.Node | null = notReturnStmts[0].astNode;
            while (p && p !== this.astRoot) {
                if (ts.isTryStatement(p) || ts.isSwitchStatement(p)) {
                    TryOrSwitchExit = true;
                    break;
                }
                p = p.parent;
            }
        }
        if (notReturnStmts.length === 1 && !(notReturnStmts[0] instanceof ConditionStatementBuilder) && !TryOrSwitchExit) {
            const notReturnStmt = notReturnStmts[0];
            notReturnStmt.next = returnStatement;
            returnStatement.lasts = new Set([notReturnStmt]);
            returnStatement.next = this.exit;
            const lasts = [...this.exit.lasts];
            lasts[lasts.indexOf(notReturnStmt)] = returnStatement;
            this.exit.lasts = new Set(lasts);
            notReturnStmt.block?.stmts.push(returnStatement);
            returnStatement.block = notReturnStmt.block;
        } else {
            this.addReturnBlock(returnStatement, notReturnStmts);
        }
    }

    resetWalked() {
        for (let stmt of this.statementArray) {
            stmt.walked = false;
        }
    }

    addStmtBuilderPosition() {
        for (const stmt of this.statementArray) {
            if (stmt.astNode) {
                const { line, character } = ts.getLineAndCharacterOfPosition(
                    this.sourceFile,
                    stmt.astNode.getStart(this.sourceFile),
                );
                stmt.line = line + 1;
                stmt.column = character + 1;
            }
        }
    }

    CfgBuilder2Array(stmt: StatementBuilder) {

        if (stmt.walked)
            return;
        stmt.walked = true;
        stmt.index = this.statementArray.length;
        if (!stmt.type.includes(' exit'))
            this.statementArray.push(stmt);
        if (stmt.type === 'ifStatement' || stmt.type === 'loopStatement' || stmt.type === 'catchOrNot') {
            let cstm = stmt as ConditionStatementBuilder;
            if (cstm.nextT == null || cstm.nextF == null) {
                this.errorTest(cstm);
                return;
            }
            this.CfgBuilder2Array(cstm.nextF);
            this.CfgBuilder2Array(cstm.nextT);
        } else if (stmt.type === 'switchStatement') {
            let sstm = stmt as SwitchStatementBuilder;
            for (let ss of sstm.nexts) {
                this.CfgBuilder2Array(ss);
            }
        } else if (stmt.type === 'tryStatement') {
            let trystm = stmt as TryStatementBuilder;
            if (trystm.tryFirst) {
                this.CfgBuilder2Array(trystm.tryFirst);
            }
            if (trystm.catchStatement) {
                this.CfgBuilder2Array(trystm.catchStatement);
            }
            if (trystm.finallyStatement) {
                this.CfgBuilder2Array(trystm.finallyStatement);
            }
            if (trystm.next) {
                this.CfgBuilder2Array(trystm.next);
            }
        } else {
            if (stmt.next != null)
                this.CfgBuilder2Array(stmt.next);
        }
    }

    getDotEdges(stmt: StatementBuilder) {
        if (this.statementArray.length === 0)
            this.CfgBuilder2Array(this.entry);
        if (stmt.walked)
            return;
        stmt.walked = true;
        if (stmt.type === 'ifStatement' || stmt.type === 'loopStatement' || stmt.type === 'catchOrNot') {
            let cstm = stmt as ConditionStatementBuilder;
            if (cstm.nextT == null || cstm.nextF == null) {
                this.errorTest(cstm);
                return;
            }
            let edge = [cstm.index, cstm.nextF.index];
            this.dotEdges.push(edge);
            edge = [cstm.index, cstm.nextT.index];
            this.dotEdges.push(edge);
            this.getDotEdges(cstm.nextF);
            this.getDotEdges(cstm.nextT);
        } else if (stmt.type === 'switchStatement') {
            let sstm = stmt as SwitchStatementBuilder;
            for (let ss of sstm.nexts) {
                let edge = [sstm.index, ss.index];
                this.dotEdges.push(edge);
                this.getDotEdges(ss);
            }
        } else {
            if (stmt.next != null) {
                let edge = [stmt.index, stmt.next.index];
                this.dotEdges.push(edge);
                this.getDotEdges(stmt.next);
            }
        }
    }

    errorTest(stmt: StatementBuilder) {
        let mes = 'ifnext error    ';
        if (this.declaringClass?.getDeclaringArkFile()) {
            mes += this.declaringClass?.getDeclaringArkFile().getName() + '.' + this.declaringClass.getName() + '.' + this.name;
        }
        mes += '\n' + stmt.code;
        throw new textError(mes);
    }

    printBlocks(): string {
        let text = '';
        if (this.declaringClass?.getDeclaringArkFile()) {
            text += this.declaringClass.getDeclaringArkFile().getName() + '\n';
        }
        for (let bi = 0; bi < this.blocks.length; bi++) {
            let block = this.blocks[bi];
            if (bi !== 0)
                text += 'label' + block.id + ':\n';
            let length = block.stmts.length;
            for (let i = 0; i < length; i++) {
                let stmt = block.stmts[i];
                if (stmt.type === 'ifStatement' || stmt.type === 'loopStatement' || stmt.type === 'catchOrNot') {
                    let cstm = stmt as ConditionStatementBuilder;
                    if (cstm.nextT == null || cstm.nextF == null) {
                        this.errorTest(cstm);
                        return text;
                    }
                    if (!cstm.nextF.block || !cstm.nextT.block) {
                        this.errorTest(cstm);
                        return text;
                    }
                    stmt.code = 'if !(' + cstm.condition + ') goto label' + cstm.nextF.block.id;
                    if (i === length - 1 && bi + 1 < this.blocks.length && this.blocks[bi + 1].id !== cstm.nextT.block.id) {
                        let gotoStm = new StatementBuilder('gotoStatement', 'goto label' + cstm.nextT.block.id, null, block.stmts[0].scopeID);
                        block.stmts.push(gotoStm);
                        length++;
                    }
                } else if (stmt.type === 'breakStatement' || stmt.type === 'continueStatement') {
                    if (!stmt.next?.block) {
                        this.errorTest(stmt);
                        return text;
                    }
                    stmt.code = 'goto label' + stmt.next?.block.id;
                } else {
                    if (i === length - 1 && stmt.next?.block && (bi + 1 < this.blocks.length && this.blocks[bi + 1].id !== stmt.next.block.id || bi + 1 === this.blocks.length)) {
                        let gotoStm = new StatementBuilder('StatementBuilder', 'goto label' + stmt.next?.block.id, null, block.stmts[0].scopeID);
                        block.stmts.push(gotoStm);
                        length++;
                    }
                }
                if (stmt.addressCode3.length === 0) {
                    text += '    ' + stmt.code + '\n';
                } else {
                    for (let ac of stmt.addressCode3) {
                        if (ac.startsWith('if') || ac.startsWith('while')) {
                            let cstm = stmt as ConditionStatementBuilder;
                            let condition = ac.substring(ac.indexOf('('));
                            let goto = '';
                            if (cstm.nextF?.block)
                                goto = 'if !' + condition + ' goto label' + cstm.nextF?.block.id;
                            stmt.addressCode3[stmt.addressCode3.indexOf(ac)] = goto;
                            text += '    ' + goto + '\n';
                        } else
                            text += '    ' + ac + '\n';
                    }
                }
            }

        }
        for (let cat of this.catches) {
            text += 'catch ' + cat.errorName + ' from label ' + cat.from + ' to label ' + cat.to + ' with label' + cat.withLabel + '\n';
        }

        return text;
    }

    buildStatementBuilder4ArrowFunction(stmt: ts.Node) {
        let s = new StatementBuilder('statement', stmt.getText(this.sourceFile), stmt, 0);
        this.entry.next = s;
        s.lasts = new Set([this.entry]);
        s.next = this.exit;
        this.exit.lasts = new Set([s]);
    }

    buildCfgBuilder() {
        let stmts: ts.Node[] = [];
        if (ts.isSourceFile(this.astRoot)) {
            stmts = [...this.astRoot.statements];
        } else if (ts.isFunctionDeclaration(this.astRoot) || ts.isMethodDeclaration(this.astRoot) || ts.isConstructorDeclaration(this.astRoot)
            || ts.isGetAccessorDeclaration(this.astRoot) || ts.isSetAccessorDeclaration(this.astRoot) || ts.isFunctionExpression(this.astRoot)) {
            if (this.astRoot.body) {
                stmts = [...this.astRoot.body.statements];
            } else {
                this.emptyBody = true;
            }
        } else if (ts.isArrowFunction(this.astRoot)) {
            if (ts.isBlock(this.astRoot.body)) {
                stmts = [...this.astRoot.body.statements];
            }
        } else if (ts.isMethodSignature(this.astRoot) || ts.isConstructSignatureDeclaration(this.astRoot)
            || ts.isCallSignatureDeclaration(this.astRoot) || ts.isFunctionTypeNode(this.astRoot)) {
            this.emptyBody = true;
        } else if (ts.isModuleDeclaration(this.astRoot) && ts.isModuleBlock(this.astRoot.body!)) {
            stmts = [...this.astRoot.body.statements];
        }
        if (!ModelUtils.isArkUIBuilderMethod(this.declaringMethod)) {
            this.walkAST(this.entry, this.exit, stmts);
        } else {
            this.handleBuilder(stmts);
        }
        if (ts.isArrowFunction(this.astRoot) && !ts.isBlock(this.astRoot.body)) {
            this.buildStatementBuilder4ArrowFunction(this.astRoot.body);
        }
        this.addReturnInEmptyMethod();
        this.deleteExit();
        this.CfgBuilder2Array(this.entry);
        this.addStmtBuilderPosition();
        this.buildBlocks();
        this.blocks = this.blocks.filter((b) => b.stmts.length !== 0);
        this.buildBlocksNextLast();
        this.addReturnStmt();
    }

    private handleBuilder(stmts: ts.Node[]): void {
        let lastStmt = this.entry;
        for (const stmt of stmts) {
            const stmtBuilder = new StatementBuilder('statement', stmt.getText(this.sourceFile), stmt, 0);
            lastStmt.next = stmtBuilder;
            stmtBuilder.lasts.add(lastStmt);
            lastStmt = stmtBuilder;
        }
        lastStmt.next = this.exit;
        this.exit.lasts.add(lastStmt);
    }

    public isBodyEmpty(): boolean {
        return this.emptyBody;
    }

    public buildCfgAndOriginalCfg(): {
        cfg: Cfg,
        locals: Set<Local>,
        globals: Map<string, GlobalRef> | null,
        aliasTypeMap: Map<string, [AliasType, AliasTypeDeclaration]>,
        traps: Trap[],
    } {
        if (ts.isArrowFunction(this.astRoot) && !ts.isBlock(this.astRoot.body)) {
            return this.buildCfgAndOriginalCfgForSimpleArrowFunction();
        }

        return this.buildNormalCfgAndOriginalCfg();
    }

    public buildCfgAndOriginalCfgForSimpleArrowFunction(): {
        cfg: Cfg,
        locals: Set<Local>,
        globals: Map<string, GlobalRef> | null,
        aliasTypeMap: Map<string, [AliasType, AliasTypeDeclaration]>,
        traps: Trap[],
    } {
        const stmts: Stmt[] = [];
        const arkIRTransformer = new ArkIRTransformer(this.sourceFile, this.declaringMethod);
        stmts.push(...arkIRTransformer.prebuildStmts());
        const expressionBodyNode = (this.astRoot as ts.ArrowFunction).body as ts.Expression;
        const expressionBodyStmts: Stmt[] = [];
        let {
            value: expressionBodyValue,
            valueOriginalPositions: expressionBodyPositions,
            stmts: tempStmts,
        } = arkIRTransformer.tsNodeToValueAndStmts(expressionBodyNode);
        expressionBodyStmts.push(...tempStmts);
        if (IRUtils.moreThanOneAddress(expressionBodyValue)) {
            ({
                value: expressionBodyValue,
                valueOriginalPositions: expressionBodyPositions,
                stmts: tempStmts,
            } = arkIRTransformer.generateAssignStmtForValue(expressionBodyValue, expressionBodyPositions));
            expressionBodyStmts.push(...tempStmts);
        }
        const returnStmt = new ArkReturnStmt(expressionBodyValue);
        returnStmt.setOperandOriginalPositions([expressionBodyPositions[0], ...expressionBodyPositions]);
        expressionBodyStmts.push(returnStmt);
        arkIRTransformer.mapStmtsToTsStmt(expressionBodyStmts, expressionBodyNode);
        stmts.push(...expressionBodyStmts);
        const cfg = new Cfg();
        const blockInCfg = new BasicBlock();
        blockInCfg.setId(0);
        stmts.forEach(stmt => {
            blockInCfg.addStmt(stmt);
            stmt.setCfg(cfg);
        });
        cfg.addBlock(blockInCfg);
        cfg.setStartingStmt(stmts[0]);
        return {
            cfg: cfg,
            locals: arkIRTransformer.getLocals(),
            globals: arkIRTransformer.getGlobals(),
            aliasTypeMap: arkIRTransformer.getAliasTypeMap(),
            traps: [],
        };
    }

    public buildNormalCfgAndOriginalCfg(): {
        cfg: Cfg,
        locals: Set<Local>,
        globals: Map<string, GlobalRef> | null,
        aliasTypeMap: Map<string, [AliasType, AliasTypeDeclaration]>,
        traps: Trap[],
    } {
        const blockBuilderToCfgBlock = new Map<Block, BasicBlock>();
        const basicBlockSet = new Set<BasicBlock>();
        const arkIRTransformer = new ArkIRTransformer(this.sourceFile, this.declaringMethod);
        const blocksContainLoopCondition = new Set<Block>();
        const blockBuildersBeforeTry = new Set<Block>();
        const blockBuildersContainSwitch: Block[] = [];
        const valueAndStmtsOfSwitchAndCasesAll: ValueAndStmts[][] = [];
        for (let i = 0; i < this.blocks.length; i++) {
            // build block in Cfg
            const stmtsInBlock: Stmt[] = [];
            if (i === 0) {
                stmtsInBlock.push(...arkIRTransformer.prebuildStmts());
            }
            const stmtsCnt = this.blocks[i].stmts.length;
            if (this.blocks[i].stmts[stmtsCnt - 1].type === 'tryStatement') {
                blockBuildersBeforeTry.add(this.blocks[i]);
            }
            for (const statementBuilder of this.blocks[i].stmts) {
                if (statementBuilder.type === 'loopStatement') {
                    blocksContainLoopCondition.add(this.blocks[i]);
                } else if (statementBuilder instanceof SwitchStatementBuilder) {
                    blockBuildersContainSwitch.push(this.blocks[i]);
                    const valueAndStmtsOfSwitchAndCases = arkIRTransformer.switchStatementToValueAndStmts(
                        statementBuilder.astNode as ts.SwitchStatement);
                    valueAndStmtsOfSwitchAndCasesAll.push(valueAndStmtsOfSwitchAndCases);
                    continue;
                }
                if (statementBuilder.astNode && statementBuilder.code !== '') {
                    arkIRTransformer.tsNodeToStmts(statementBuilder.astNode).forEach(s => stmtsInBlock.push(s));
                } else if (statementBuilder.code.startsWith('return')) {
                    stmtsInBlock.push(new ArkReturnVoidStmt());
                }
            }
            const blockInCfg = new BasicBlock();
            blockInCfg.setId(this.blocks[i].id);
            for (const stmt of stmtsInBlock) {
                blockInCfg.addStmt(stmt);
            }
            basicBlockSet.add(blockInCfg);
            blockBuilderToCfgBlock.set(this.blocks[i], blockInCfg);
        }
        let currBlockId = this.blocks.length;
        this.linkBasicBlocks(blockBuilderToCfgBlock);
        this.rebuildBlocksInLoop(blockBuilderToCfgBlock, blocksContainLoopCondition, basicBlockSet);
        this.buildSwitch(blockBuilderToCfgBlock, blockBuildersContainSwitch, valueAndStmtsOfSwitchAndCasesAll,
            arkIRTransformer, basicBlockSet);
        this.rebuildBlocksContainConditionalOperator(basicBlockSet);
        const traps = this.buildTraps(blockBuilderToCfgBlock, blockBuildersBeforeTry, arkIRTransformer, basicBlockSet);
        for (const blockBuilder of this.blocks) {
            if (blockBuilder.id === -1) {
                blockBuilder.id = currBlockId++;
                const block = blockBuilderToCfgBlock.get(blockBuilder) as BasicBlock;
                block.setId(blockBuilder.id);
            }
        }
        const cfg = new Cfg();
        const startingBasicBlock = blockBuilderToCfgBlock.get(this.blocks[0])!;
        cfg.setStartingStmt(startingBasicBlock.getStmts()[0]);
        currBlockId = 0;
        for (const basicBlock of basicBlockSet) {
            basicBlock.setId(currBlockId++);
            cfg.addBlock(basicBlock);
        }
        for (const stmt of cfg.getStmts()) {
            stmt.setCfg(cfg);
        }

        return {
            cfg: cfg,
            locals: arkIRTransformer.getLocals(),
            globals: arkIRTransformer.getGlobals(),
            aliasTypeMap: arkIRTransformer.getAliasTypeMap(),
            traps: traps,
        };
    }

    private linkBasicBlocks(blockBuilderToCfgBlock: Map<Block, BasicBlock>): void {
        for (const [blockBuilder, cfgBlock] of blockBuilderToCfgBlock) {
            for (const successorBlockBuilder of blockBuilder.nexts) {
                if (!blockBuilderToCfgBlock.get(successorBlockBuilder)) {
                    continue;
                }
                const successorBlock = blockBuilderToCfgBlock.get(successorBlockBuilder) as BasicBlock;
                cfgBlock.addSuccessorBlock(successorBlock);
            }
            for (const predecessorBlockBuilder of blockBuilder.lasts) {
                if (!blockBuilderToCfgBlock.get(predecessorBlockBuilder)) {
                    continue;
                }
                const predecessorBlock = blockBuilderToCfgBlock.get(predecessorBlockBuilder) as BasicBlock;
                cfgBlock.addPredecessorBlock(predecessorBlock);
            }
        }
    }

    private rebuildBlocksContainConditionalOperator(basicBlockSet: Set<BasicBlock>): void {
        if (ModelUtils.isArkUIBuilderMethod(this.declaringMethod)) {
            this.deleteDummyConditionalOperatorStmt(basicBlockSet);
            return;
        }

        const currBasicBlocks = Array.from(basicBlockSet);
        for (const currBasicBlock of currBasicBlocks) {
            const stmtsInCurrBasicBlock = Array.from(currBasicBlock.getStmts());
            const stmtsCnt = stmtsInCurrBasicBlock.length;
            let conditionalOperatorEndPos = -1;
            for (let i = stmtsCnt - 1; i >= 0; i--) {
                const stmt = stmtsInCurrBasicBlock[i];
                if (stmt instanceof DummyStmt && stmt.toString()
                    ?.startsWith(ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR_END_STMT)) {
                    conditionalOperatorEndPos = i;
                    break;
                }
            }
            if (conditionalOperatorEndPos === -1) {
                continue;
            }

            let {
                generatedTopBlock: generatedTopBlock,
                generatedBottomBlocks: generatedBottomBlocks,
            } = this.generateBlocksContainConditionalOperatorGroup(
                stmtsInCurrBasicBlock.slice(0, conditionalOperatorEndPos + 1), basicBlockSet);

            if (conditionalOperatorEndPos !== stmtsCnt - 1) { // need create a new basic block for rest statements
                const {
                    generatedTopBlock: extraBlock,
                } = this.generateBlockWithoutConditionalOperator(
                    stmtsInCurrBasicBlock.slice(conditionalOperatorEndPos + 1));
                generatedBottomBlocks.forEach(generatedBottomBlock => {
                    generatedBottomBlock.addSuccessorBlock(extraBlock);
                    extraBlock.addPredecessorBlock(generatedBottomBlock);
                });
                basicBlockSet.add(extraBlock);
                generatedBottomBlocks = this.removeUnnecessaryBlocksInConditionalOperator(extraBlock, basicBlockSet);
            }
            this.relinkPrevAndSuccOfBlockContainConditionalOperator(currBasicBlock, generatedTopBlock, generatedBottomBlocks);
            basicBlockSet.delete(currBasicBlock);
        }
    }

    private relinkPrevAndSuccOfBlockContainConditionalOperator(currBasicBlock: BasicBlock,
                                                               generatedTopBlock: BasicBlock,
                                                               generatedBottomBlocks: BasicBlock[]): void {
        const predecessorsOfCurrBasicBlock = Array.from(currBasicBlock.getPredecessors());
        predecessorsOfCurrBasicBlock.forEach(predecessor => {
            predecessor.removeSuccessorBlock(currBasicBlock);
            currBasicBlock.removePredecessorBlock(predecessor);
            generatedTopBlock.addPredecessorBlock(predecessor);
            predecessor.addSuccessorBlock(generatedTopBlock);
        });
        const successorsOfCurrBasicBlock = Array.from(currBasicBlock.getSuccessors());
        successorsOfCurrBasicBlock.forEach(successor => {
            successor.removePredecessorBlock(currBasicBlock);
            currBasicBlock.removeSuccessorBlock(successor);
            generatedBottomBlocks.forEach(generatedBottomBlock => {
                generatedBottomBlock.addSuccessorBlock(successor);
                successor.addPredecessorBlock(generatedBottomBlock);
            });
        });
    }

    private generateBlocksContainConditionalOperatorGroup(sourceStmts: Stmt[], basicBlockSet: Set<BasicBlock>): {
        generatedTopBlock: BasicBlock,
        generatedBottomBlocks: BasicBlock[],
    } {
        const { firstEndPos: firstEndPos } = this.findFirstConditionalOperator(sourceStmts);
        if (firstEndPos === -1) {
            return this.generateBlockWithoutConditionalOperator(sourceStmts);
        }
        const {
            generatedTopBlock: firstGeneratedTopBlock,
            generatedBottomBlocks: firstGeneratedBottomBlocks,
            generatedAllBlocks: firstGeneratedAllBlocks,
        } = this.generateBlocksContainSingleConditionalOperator(sourceStmts.slice(0, firstEndPos + 1));
        const generatedTopBlock = firstGeneratedTopBlock;
        let generatedBottomBlocks = firstGeneratedBottomBlocks;
        firstGeneratedAllBlocks.forEach(block => basicBlockSet.add(block));
        const stmtsCnt = sourceStmts.length;
        if (firstEndPos !== stmtsCnt - 1) { // need handle other conditional operators
            const {
                generatedTopBlock: restGeneratedTopBlock,
                generatedBottomBlocks: restGeneratedBottomBlocks,
            } = this.generateBlocksContainConditionalOperatorGroup(
                sourceStmts.slice(firstEndPos + 1, stmtsCnt), basicBlockSet);
            firstGeneratedBottomBlocks.forEach(firstGeneratedBottomBlock => {
                firstGeneratedBottomBlock.addSuccessorBlock(restGeneratedTopBlock);
                restGeneratedTopBlock.addPredecessorBlock(firstGeneratedBottomBlock);
            });
            restGeneratedBottomBlocks.forEach(block => basicBlockSet.add(block));
            this.removeUnnecessaryBlocksInConditionalOperator(restGeneratedTopBlock, basicBlockSet);
            generatedBottomBlocks = restGeneratedBottomBlocks;
        }
        return { generatedTopBlock, generatedBottomBlocks };
    }

    private generateBlocksContainSingleConditionalOperator(sourceStmts: Stmt[]): {
        generatedTopBlock: BasicBlock,
        generatedBottomBlocks: BasicBlock[],
        generatedAllBlocks: BasicBlock[],
    } {
        const {
            firstIfTruePos: ifTruePos,
            firstIfFalsePos: ifFalsePos,
            firstEndPos: endPos,
        } = this.findFirstConditionalOperator(sourceStmts);
        if (endPos === -1) {
            return this.generateBlockWithoutConditionalOperator(sourceStmts);
        }
        const {
            generatedTopBlock: generatedTopBlock,
            generatedAllBlocks: generatedAllBlocks,
        } = this.generateBlockWithoutConditionalOperator(sourceStmts.slice(0, ifTruePos));
        let generatedBottomBlocks: BasicBlock[] = [];
        const {
            generatedTopBlock: generatedTopBlockOfTrueBranch,
            generatedBottomBlocks: generatedBottomBlocksOfTrueBranch,
            generatedAllBlocks: generatedAllBlocksOfTrueBranch,
        } = this.generateBlocksContainSingleConditionalOperator(sourceStmts.slice(ifTruePos + 1, ifFalsePos));
        generatedBottomBlocks.push(...generatedBottomBlocksOfTrueBranch);
        generatedAllBlocks.push(...generatedAllBlocksOfTrueBranch);
        const {
            generatedTopBlock: generatedTopBlockOfFalseBranch,
            generatedBottomBlocks: generatedBottomBlocksOfFalseBranch,
            generatedAllBlocks: generatedAllBlocksOfFalseBranch,
        } = this.generateBlocksContainSingleConditionalOperator(sourceStmts.slice(ifFalsePos + 1, endPos));
        generatedBottomBlocks.push(...generatedBottomBlocksOfFalseBranch);
        generatedAllBlocks.push(...generatedAllBlocksOfFalseBranch);

        generatedTopBlock.addSuccessorBlock(generatedTopBlockOfTrueBranch);
        generatedTopBlockOfTrueBranch.addPredecessorBlock(generatedTopBlock);
        generatedTopBlock.addSuccessorBlock(generatedTopBlockOfFalseBranch);
        generatedTopBlockOfFalseBranch.addPredecessorBlock(generatedTopBlock);
        const stmtsCnt = sourceStmts.length;
        if (endPos !== stmtsCnt - 1) { // need create a new basic block for rest statements
            const {
                generatedTopBlock: extraBlock,
            } = this.generateBlockWithoutConditionalOperator(sourceStmts.slice(endPos + 1));
            generatedBottomBlocks.forEach(generatedBottomBlock => {
                generatedBottomBlock.addSuccessorBlock(extraBlock);
                extraBlock.addPredecessorBlock(generatedBottomBlock);
            });
            generatedBottomBlocks = [extraBlock];
            generatedAllBlocks.push(extraBlock);
        }
        return { generatedTopBlock, generatedBottomBlocks, generatedAllBlocks };
    }

    private generateBlockWithoutConditionalOperator(sourceStmts: Stmt[]): {
        generatedTopBlock: BasicBlock,
        generatedBottomBlocks: BasicBlock[],
        generatedAllBlocks: BasicBlock[],
    } {
        const generatedBlock = new BasicBlock();
        sourceStmts.forEach(stmt => generatedBlock.addStmt(stmt));
        return {
            generatedTopBlock: generatedBlock,
            generatedBottomBlocks: [generatedBlock],
            generatedAllBlocks: [generatedBlock],
        };
    }

    private deleteDummyConditionalOperatorStmt(basicBlockSet: Set<BasicBlock>): void {
        for (const basicBlock of basicBlockSet) {
            const stmts = Array.from(basicBlock.getStmts());
            for (const stmt of stmts) {
                if (stmt instanceof DummyStmt && stmt.toString()
                    ?.startsWith(ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR)) {
                    basicBlock.remove(stmt);
                }
            }
        }
    }

    private findFirstConditionalOperator(stmts: Stmt[]): {
        firstIfTruePos: number, firstIfFalsePos: number, firstEndPos: number,
    } {
        let firstIfTruePos = -1;
        let firstIfFalsePos = -1;
        let firstEndPos = -1;
        let firstConditionalOperatorNo = '';
        for (let i = 0; i < stmts.length; i++) {
            const stmt = stmts[i];
            if (stmt instanceof DummyStmt) {
                if (stmt.toString()
                    .startsWith(ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR_IF_TRUE_STMT) && firstIfTruePos === -1) {
                    firstIfTruePos = i;
                    firstConditionalOperatorNo =
                        stmt.toString().replace(ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR_IF_TRUE_STMT, '');
                } else if (stmt.toString() === ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR_IF_FALSE_STMT + firstConditionalOperatorNo) {
                    firstIfFalsePos = i;
                } else if (stmt.toString() === ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR_END_STMT + firstConditionalOperatorNo) {
                    firstEndPos = i;
                }
            }
        }
        return { firstIfTruePos, firstIfFalsePos, firstEndPos };
    }

    private removeUnnecessaryBlocksInConditionalOperator(bottomBlock: BasicBlock, allBlocks: Set<BasicBlock>): BasicBlock[] {
        const firstStmtInBottom = bottomBlock.getStmts()[0];
        if (!(firstStmtInBottom instanceof ArkAssignStmt)) {
            return [bottomBlock];
        }

        const targetValue = firstStmtInBottom.getLeftOp();
        const tempResultValue = firstStmtInBottom.getRightOp();
        if (!(targetValue instanceof Local && IRUtils.isTempLocal(tempResultValue))) {
            return [bottomBlock];
        }
        const oldPredecessors = Array.from(bottomBlock.getPredecessors());
        const newPredecessors: BasicBlock[] = [];
        for (const predecessor of oldPredecessors) {
            predecessor.removeSuccessorBlock(bottomBlock);
            newPredecessors.push(
                ...this.replaceTempRecursively(predecessor, targetValue as Local, tempResultValue as Local,
                    allBlocks));
        }

        bottomBlock.remove(firstStmtInBottom);
        if (bottomBlock.getStmts().length === 0) { // must be a new block without successors
            allBlocks.delete(bottomBlock);
            return newPredecessors;
        }

        oldPredecessors.forEach((oldPredecessor) => {
            bottomBlock.removePredecessorBlock(oldPredecessor);
        });
        newPredecessors.forEach((newPredecessor) => {
            bottomBlock.addPredecessorBlock(newPredecessor);
            newPredecessor.addSuccessorBlock(bottomBlock);
        });
        return [bottomBlock];
    }

    private replaceTempRecursively(currBottomBlock: BasicBlock, targetLocal: Local,
                                   tempResultLocal: Local, allBlocks: Set<BasicBlock>): BasicBlock[] {
        const stmts = currBottomBlock.getStmts();
        const stmtsCnt = stmts.length;
        let tempResultReassignStmt: Stmt | null = null;
        for (let i = stmtsCnt - 1; i >= 0; i--) {
            const stmt = stmts[i];
            if (stmt instanceof ArkAssignStmt && stmt.getLeftOp() === tempResultLocal) {
                if (IRUtils.isTempLocal(stmt.getRightOp())) {
                    tempResultReassignStmt = stmt;
                } else {
                    stmt.setLeftOp(targetLocal);
                }
            }
        }

        let newBottomBlocks: BasicBlock[] = [];
        if (tempResultReassignStmt) {
            const oldPredecessors = currBottomBlock.getPredecessors();
            const newPredecessors: BasicBlock[] = [];
            const prevTempResultLocal = (tempResultReassignStmt as ArkAssignStmt).getRightOp() as Local;
            for (const predecessor of oldPredecessors) {
                predecessor.removeSuccessorBlock(currBottomBlock);
                newPredecessors.push(
                    ...this.replaceTempRecursively(predecessor, targetLocal, prevTempResultLocal, allBlocks));
            }

            currBottomBlock.remove(tempResultReassignStmt);
            if (currBottomBlock.getStmts().length === 0) {
                // remove this block
                newBottomBlocks = newPredecessors;
                allBlocks.delete(currBottomBlock);
            } else {
                currBottomBlock.getPredecessors().splice(0, oldPredecessors.length, ...newPredecessors);
                newPredecessors.forEach((newPredecessor) => {
                    newPredecessor.addSuccessorBlock(currBottomBlock);
                });
                newBottomBlocks = [currBottomBlock];
            }
        } else {
            newBottomBlocks = [currBottomBlock];
        }
        return newBottomBlocks;
    }

    private rebuildBlocksInLoop(blockBuilderToCfgBlock: Map<Block, BasicBlock>, blocksContainLoopCondition: Set<Block>,
                                basicBlockSet: Set<BasicBlock>): void {
        for (const blockBuilder of blocksContainLoopCondition) {
            if (!blockBuilderToCfgBlock.get(blockBuilder)) {
                continue;
            }
            const block = blockBuilderToCfgBlock.get(blockBuilder) as BasicBlock;
            const blockId = block.getId();
            const stmts = block.getStmts();
            const stmtsCnt = stmts.length;
            const { ifStmtIdx, iteratorNextStmtIdx, dummyInitializerStmtIdx } = this.findIteratorIdx(stmts);
            if (iteratorNextStmtIdx !== -1 || dummyInitializerStmtIdx !== -1) {
                const lastStmtIdxBeforeCondition = iteratorNextStmtIdx !== -1 ? iteratorNextStmtIdx : dummyInitializerStmtIdx;
                const stmtsInsertBeforeCondition = stmts.slice(0, lastStmtIdxBeforeCondition);

                let prevBlockBuilderContainsLoop = this.doesPrevBlockBuilderContainLoop(blockBuilder, blockId,
                    blocksContainLoopCondition);
                if (prevBlockBuilderContainsLoop) {
                    // should create an extra block when previous block contains loop condition
                    this.insertBeforeConditionBlockBuilder(blockBuilderToCfgBlock, blockBuilder,
                        stmtsInsertBeforeCondition, false, basicBlockSet);
                } else {
                    const blockBuilderBeforeCondition = blockBuilder.lasts[0];
                    const blockBeforeCondition = blockBuilderToCfgBlock.get(blockBuilderBeforeCondition) as BasicBlock;
                    blockBeforeCondition?.getStmts().push(...stmtsInsertBeforeCondition);
                }
                if (dummyInitializerStmtIdx !== -1 && ifStmtIdx !== stmtsCnt - 1) {
                    // put incrementor statements into block which reenters condition
                    this.adjustIncrementorStmts(stmts, ifStmtIdx, blockBuilder, blockId, blockBuilderToCfgBlock,
                        blocksContainLoopCondition, basicBlockSet);
                } else if (iteratorNextStmtIdx !== -1) {
                    // put statements which get value of iterator into block after condition
                    const blockBuilderAfterCondition = blockBuilder.nexts[0];
                    const blockAfterCondition = blockBuilderToCfgBlock.get(blockBuilderAfterCondition) as BasicBlock;
                    const stmtsAfterCondition = stmts.slice(ifStmtIdx + 1);
                    blockAfterCondition?.getStmts().splice(0, 0, ...stmtsAfterCondition);
                }
                // remove statements which should not in condition
                const firstStmtIdxInCondition = iteratorNextStmtIdx !== -1 ? iteratorNextStmtIdx : dummyInitializerStmtIdx + 1;
                stmts.splice(0, firstStmtIdxInCondition);
                stmts.splice(ifStmtIdx - firstStmtIdxInCondition + 1);
            }
        }
    }

    private doesPrevBlockBuilderContainLoop(currBlockBuilder: Block, currBlockId: number,
                                            blocksContainLoopCondition: Set<Block>): boolean {
        let prevBlockBuilderContainsLoop = false;
        for (const prevBlockBuilder of currBlockBuilder.lasts) {
            if (prevBlockBuilder.id < currBlockId && blocksContainLoopCondition.has(prevBlockBuilder)) {
                prevBlockBuilderContainsLoop = true;
                break;
            }
        }
        return prevBlockBuilderContainsLoop;
    }

    private insertBeforeConditionBlockBuilder(blockBuilderToCfgBlock: Map<Block, BasicBlock>,
                                              conditionBlockBuilder: Block, stmtsInsertBeforeCondition: Stmt[],
                                              collectReenter: Boolean, basicBlockSet: Set<BasicBlock>): void {
        const blockId = conditionBlockBuilder.id;
        const block = blockBuilderToCfgBlock.get(conditionBlockBuilder) as BasicBlock;
        const blockBuildersBeforeCondition: Block[] = [];
        const blocksBeforeCondition: BasicBlock[] = [];
        const blockBuildersReenterCondition: Block[] = [];
        const blocksReenterCondition: BasicBlock[] = [];
        for (const prevBlockBuilder of conditionBlockBuilder.lasts) {
            const prevBlock = blockBuilderToCfgBlock.get(prevBlockBuilder) as BasicBlock;
            if (prevBlock.getId() < blockId) {
                blockBuildersBeforeCondition.push(prevBlockBuilder);
                blocksBeforeCondition.push(prevBlock);
            } else {
                blockBuildersReenterCondition.push(prevBlockBuilder);
                blocksReenterCondition.push(prevBlock);
            }
        }

        let collectedBlockBuilders: Block[] = [];
        let collectedBlocks: BasicBlock[] = [];
        if (collectReenter) {
            collectedBlockBuilders = blockBuildersReenterCondition;
            collectedBlocks = blocksReenterCondition;
        } else {
            collectedBlockBuilders = blockBuildersBeforeCondition;
            collectedBlocks = blocksBeforeCondition;
        }

        const blockBuilderInsertBeforeCondition = new Block(-1, []);
        blockBuilderInsertBeforeCondition.lasts.push(...collectedBlockBuilders);
        blockBuilderInsertBeforeCondition.nexts.push(conditionBlockBuilder);
        const blockInsertBeforeCondition = new BasicBlock();
        blockInsertBeforeCondition.getStmts().push(...stmtsInsertBeforeCondition);
        blockInsertBeforeCondition.getPredecessors().push(...collectedBlocks);
        blockInsertBeforeCondition.addSuccessorBlock(block);

        for (const prevBlockBuilder of collectedBlockBuilders) {
            const prevBlock = blockBuilderToCfgBlock.get(prevBlockBuilder) as BasicBlock;
            for (let j = 0; j < prevBlockBuilder.nexts.length; j++) {
                if (prevBlockBuilder.nexts[j] === conditionBlockBuilder) {
                    prevBlockBuilder.nexts[j] = blockBuilderInsertBeforeCondition;
                    prevBlock.setSuccessorBlock(j, blockInsertBeforeCondition);
                    break;
                }
            }
        }

        let newPrevBlockBuildersBeforeCondition: Block[] = [];
        let newPrevBlocksBeforeCondition: BasicBlock[] = [];
        if (collectReenter) {
            newPrevBlockBuildersBeforeCondition = [...blockBuildersBeforeCondition, blockBuilderInsertBeforeCondition];
            newPrevBlocksBeforeCondition = [...blocksBeforeCondition, blockInsertBeforeCondition];
        } else {
            newPrevBlockBuildersBeforeCondition = [blockBuilderInsertBeforeCondition, ...blockBuildersReenterCondition];
            newPrevBlocksBeforeCondition = [blockInsertBeforeCondition, ...blocksReenterCondition];
        }

        conditionBlockBuilder.lasts = newPrevBlockBuildersBeforeCondition;
        const predecessorsCnt = block.getPredecessors().length;
        block.getPredecessors().splice(0, predecessorsCnt, ...newPrevBlocksBeforeCondition);

        this.blocks.push(blockBuilderInsertBeforeCondition);
        basicBlockSet.add(blockInsertBeforeCondition);
        blockBuilderToCfgBlock.set(blockBuilderInsertBeforeCondition, blockInsertBeforeCondition);
    }

    private findIteratorIdx(stmts: Stmt[]): {
        ifStmtIdx: number, iteratorNextStmtIdx: number, dummyInitializerStmtIdx: number
    } {
        let ifStmtIdx = -1;
        let iteratorNextStmtIdx = -1;
        let dummyInitializerStmtIdx = -1;
        const stmtsCnt = stmts.length;
        for (let i = 0; i < stmtsCnt; i++) {
            const stmt = stmts[i];
            if (stmt instanceof ArkAssignStmt && stmt.getRightOp() instanceof AbstractInvokeExpr) {
                const invokeExpr = stmt.getRightOp() as AbstractInvokeExpr;
                if (invokeExpr.getMethodSignature().getMethodSubSignature()
                    .getMethodName() === Builtin.ITERATOR_NEXT) {
                    iteratorNextStmtIdx = i;
                    continue;
                }
            }
            if (stmt.toString() === ArkIRTransformer.DUMMY_LOOP_INITIALIZER_STMT) {
                dummyInitializerStmtIdx = i;
                continue;
            }
            if (stmt instanceof ArkIfStmt) {
                ifStmtIdx = i;
                break;
            }
        }
        return {
            ifStmtIdx: ifStmtIdx,
            iteratorNextStmtIdx: iteratorNextStmtIdx,
            dummyInitializerStmtIdx: dummyInitializerStmtIdx,
        };
    }

    private adjustIncrementorStmts(stmts: Stmt[], ifStmtIdx: number, currBlockBuilder: Block, currBlockId: number,
                                   blockBuilderToCfgBlock: Map<Block, BasicBlock>,
                                   blocksContainLoopCondition: Set<Block>, basicBlockSet: Set<BasicBlock>): void {
        const stmtsReenterCondition = stmts.slice(ifStmtIdx + 1);
        const blockBuildersReenterCondition: Block[] = [];
        for (const prevBlockBuilder of currBlockBuilder.lasts) {
            const prevBlock = blockBuilderToCfgBlock.get(prevBlockBuilder) as BasicBlock;

            if (prevBlock.getId() > currBlockId) {
                blockBuildersReenterCondition.push(prevBlockBuilder);
            }
        }

        if (blockBuildersReenterCondition.length > 1 || blocksContainLoopCondition.has(
            blockBuildersReenterCondition[0])) {
            // put incrementor statements into an extra block
            this.insertBeforeConditionBlockBuilder(blockBuilderToCfgBlock, currBlockBuilder,
                stmtsReenterCondition, true, basicBlockSet);
        } else {
            // put incrementor statements into prev reenter block
            const blockReenterCondition = blockBuilderToCfgBlock.get(
                blockBuildersReenterCondition[0]) as BasicBlock;
            blockReenterCondition?.getStmts().push(...stmtsReenterCondition);
        }
    }

    private buildTraps(blockBuilderToCfgBlock: Map<Block, BasicBlock>, blockBuildersBeforeTry: Set<Block>,
                       arkIRTransformer: ArkIRTransformer, basicBlockSet: Set<BasicBlock>): Trap[] {
        const traps: Trap[] = [];
        for (const blockBuilderBeforeTry of blockBuildersBeforeTry) {
            if (blockBuilderBeforeTry.nexts.length === 0) {
                logger.error(`can't find try block.`);
                continue;
            }
            const blockBuilderContainTry = blockBuilderBeforeTry.nexts[0];
            const stmtsCnt = blockBuilderBeforeTry.stmts.length;
            const tryStmtBuilder = blockBuilderBeforeTry.stmts[stmtsCnt - 1] as TryStatementBuilder;
            const finallyBlockBuilder = tryStmtBuilder.finallyStatement?.block;
            if (!finallyBlockBuilder) {
                logger.error(`can't find finally block or dummy finally block.`);
                continue;
            }
            const { bfsBlocks: tryBfsBlocks, tailBlocks: tryTailBlocks } = this.getAllBlocksBFS(blockBuilderToCfgBlock,
                blockBuilderContainTry, finallyBlockBuilder);
            let catchBfsBlocks: BasicBlock[] = [];
            let catchTailBlocks: BasicBlock[] = [];
            const catchBlockBuilder = tryStmtBuilder.catchStatement?.block;
            if (catchBlockBuilder) {
                ({ bfsBlocks: catchBfsBlocks, tailBlocks: catchTailBlocks } = this.getAllBlocksBFS(
                    blockBuilderToCfgBlock, catchBlockBuilder));
            }
            const finallyStmts = finallyBlockBuilder.stmts;
            const blockBuilderAfterFinally = tryStmtBuilder.afterFinal?.block;
            if (!blockBuilderAfterFinally) {
                logger.error(`can't find block after try...catch.`);
                continue;
            }
            if (finallyStmts.length === 1 && finallyStmts[0].code === 'dummyFinally') { // no finally block
                const trapsIfNoFinally = this.buildTrapsIfNoFinally(tryBfsBlocks, tryTailBlocks, catchBfsBlocks,
                    catchTailBlocks,
                    finallyBlockBuilder, blockBuilderAfterFinally, basicBlockSet, blockBuilderToCfgBlock);
                if (trapsIfNoFinally) {
                    traps.push(...trapsIfNoFinally);
                }
            } else {
                const trapsIfFinallyExist = this.buildTrapsIfFinallyExist(tryBfsBlocks, tryTailBlocks, catchBfsBlocks,
                    catchTailBlocks, finallyBlockBuilder, blockBuilderAfterFinally, basicBlockSet, arkIRTransformer,
                    blockBuilderToCfgBlock);
                traps.push(...trapsIfFinallyExist);
            }
        }
        return traps;
    }

    private buildTrapsIfNoFinally(tryBfsBlocks: BasicBlock[], tryTailBlocks: BasicBlock[], catchBfsBlocks: BasicBlock[],
                                  catchTailBlocks: BasicBlock[], finallyBlockBuilder: Block,
                                  blockBuilderAfterFinally: Block, basicBlockSet: Set<BasicBlock>,
                                  blockBuilderToCfgBlock: Map<Block, BasicBlock>): Trap[] | null {
        if (catchBfsBlocks.length === 0) {
            logger.error(`catch block expected.`);
            return null;
        }
        if (!blockBuilderToCfgBlock.has(blockBuilderAfterFinally)) {
            logger.error(`can't find basicBlock corresponding to the blockBuilder.`);
            return null;
        }
        let blockAfterFinally: BasicBlock = blockBuilderToCfgBlock.get(blockBuilderAfterFinally)!;
        if (!blockBuilderToCfgBlock.has(finallyBlockBuilder)) {
            logger.error(`can't find basicBlock corresponding to the blockBuilder.`);
            return null;
        }
        const finallyBlock = blockBuilderToCfgBlock.get(finallyBlockBuilder)!;
        let dummyFinallyIdxInPredecessors = -1;
        for (let i = 0; i < blockAfterFinally.getPredecessors().length; i++) {
            if (blockAfterFinally.getPredecessors()[i] === finallyBlock) {
                dummyFinallyIdxInPredecessors = i;
                break;
            }
        }
        if (dummyFinallyIdxInPredecessors === -1) {
            return null;
        }
        blockAfterFinally.getPredecessors().splice(dummyFinallyIdxInPredecessors, 1);
        for (const tryTailBlock of tryTailBlocks) {
            tryTailBlock.setSuccessorBlock(0, blockAfterFinally);
            blockAfterFinally.addPredecessorBlock(tryTailBlock);
        }
        basicBlockSet.delete(finallyBlock);

        for (const catchTailBlock of catchTailBlocks) {
            catchTailBlock.addSuccessorBlock(blockAfterFinally);
            blockAfterFinally.addPredecessorBlock(catchTailBlock);
        }
        for (const tryTailBlock of tryTailBlocks) {
            tryTailBlock.addExceptionalSuccessorBlock(catchBfsBlocks[0]);
        }
        return [new Trap(tryBfsBlocks, catchBfsBlocks)];
    }

    private buildTrapsIfFinallyExist(tryBfsBlocks: BasicBlock[], tryTailBlocks: BasicBlock[],
                                     catchBfsBlocks: BasicBlock[], catchTailBlocks: BasicBlock[],
                                     finallyBlockBuilder: Block, blockBuilderAfterFinally: Block,
                                     basicBlockSet: Set<BasicBlock>, arkIRTransformer: ArkIRTransformer,
                                     blockBuilderToCfgBlock: Map<Block, BasicBlock>): Trap[] {
        const { bfsBlocks: finallyBfsBlocks, tailBlocks: finallyTailBlocks } = this.getAllBlocksBFS(
            blockBuilderToCfgBlock,
            finallyBlockBuilder, blockBuilderAfterFinally);
        const copyFinallyBfsBlocks = this.copyFinallyBlocks(finallyBfsBlocks, finallyTailBlocks, basicBlockSet,
            arkIRTransformer, blockBuilderToCfgBlock);
        const traps: Trap[] = [];
        if (catchBfsBlocks.length !== 0) {
            for (const catchTailBlock of catchTailBlocks) {
                catchTailBlock.addSuccessorBlock(finallyBfsBlocks[0]);
                finallyBfsBlocks[0].addPredecessorBlock(catchTailBlock);
            }
            // try -> catch trap
            for (const tryTailBlock of tryTailBlocks) {
                tryTailBlock.addExceptionalSuccessorBlock(catchBfsBlocks[0]);
            }
            traps.push(new Trap(tryBfsBlocks, catchBfsBlocks));
            // catch -> finally trap
            for (const catchTailBlock of catchTailBlocks) {
                catchTailBlock.addExceptionalSuccessorBlock(copyFinallyBfsBlocks[0]);
            }
            traps.push(new Trap(catchBfsBlocks, copyFinallyBfsBlocks));
        } else {
            // try -> finally trap
            for (const tryTailBlock of tryTailBlocks) {
                tryTailBlock.addExceptionalSuccessorBlock(copyFinallyBfsBlocks[0]);
            }
            traps.push(new Trap(tryBfsBlocks, copyFinallyBfsBlocks));
        }
        return traps;
    }

    private getAllBlocksBFS(blockBuilderToCfgBlock: Map<Block, BasicBlock>, startBlockBuilder: Block,
                            endBlockBuilder?: Block): { bfsBlocks: BasicBlock[], tailBlocks: BasicBlock[] } {
        const bfsBlocks: BasicBlock[] = [];
        const tailBlocks: BasicBlock[] = [];
        const queue: Block[] = [];
        const visitedBlockBuilders = new Set<Block>();
        queue.push(startBlockBuilder);
        while (queue.length !== 0) {
            const currBlockBuilder = queue.splice(0, 1)[0];
            if (visitedBlockBuilders.has(currBlockBuilder)) {
                continue;
            }
            visitedBlockBuilders.add(currBlockBuilder);
            if (!blockBuilderToCfgBlock.has(currBlockBuilder)) {
                logger.error(`can't find basicBlock corresponding to the blockBuilder.`);
                continue;
            }
            const currBlock = blockBuilderToCfgBlock.get(currBlockBuilder)!;
            bfsBlocks.push(currBlock);

            const childList = currBlockBuilder.nexts;
            if (childList.length === 0 || (childList.length !== 0 && (childList[0] === endBlockBuilder))) {
                if (childList[0] === endBlockBuilder) {
                    tailBlocks.push(currBlock);
                    continue;
                }
            }
            if (childList.length !== 0) {
                for (const child of childList) {
                    queue.push(child);
                }
            }
        }
        return { bfsBlocks, tailBlocks };
    }

    private copyFinallyBlocks(finallyBfsBlocks: BasicBlock[], finallyTailBlocks: BasicBlock[],
                              basicBlockSet: Set<BasicBlock>, arkIRTransformer: ArkIRTransformer,
                              blockBuilderToCfgBlock: Map<Block, BasicBlock>): BasicBlock[] {
        const copyFinallyBfsBlocks = this.copyBlocks(finallyBfsBlocks);
        const caughtExceptionRef = new ArkCaughtExceptionRef(UnknownType.getInstance());
        const {
            value: exceptionValue,
            stmts: exceptionAssignStmts,
        } = arkIRTransformer.generateAssignStmtForValue(caughtExceptionRef, [FullPosition.DEFAULT]);
        copyFinallyBfsBlocks[0].addHead(exceptionAssignStmts);
        const finallyPredecessorsCnt = copyFinallyBfsBlocks[0].getPredecessors().length;
        copyFinallyBfsBlocks[0].getPredecessors().splice(0, finallyPredecessorsCnt);
        const throwStmt = new ArkThrowStmt(exceptionValue);
        let copyFinallyTailBlocks = copyFinallyBfsBlocks.splice(
            copyFinallyBfsBlocks.length - finallyTailBlocks.length, finallyTailBlocks.length);
        copyFinallyTailBlocks.forEach((copyFinallyTailBlock: BasicBlock) => {
            const successorsCnt = copyFinallyTailBlock.getSuccessors().length;
            copyFinallyTailBlock.getSuccessors().splice(0, successorsCnt);
        });
        if (copyFinallyTailBlocks.length > 1) {
            const newCopyFinallyTailBlock = new BasicBlock();
            copyFinallyTailBlocks.forEach((copyFinallyTailBlock: BasicBlock) => {
                copyFinallyTailBlock.addSuccessorBlock(newCopyFinallyTailBlock);
                newCopyFinallyTailBlock.addPredecessorBlock(copyFinallyTailBlock);
            });
            copyFinallyTailBlocks = [newCopyFinallyTailBlock];
        }
        copyFinallyTailBlocks[0]?.addStmt(throwStmt);
        copyFinallyBfsBlocks.push(...copyFinallyTailBlocks);
        copyFinallyBfsBlocks.forEach((copyFinallyBfsBlock: BasicBlock) => {
            basicBlockSet.add(copyFinallyBfsBlock);
        });
        return copyFinallyBfsBlocks;
    }

    private copyBlocks(sourceBlocks: BasicBlock[]): BasicBlock[] {
        const sourceToTarget = new Map<BasicBlock, BasicBlock>();
        const targetBlocks: BasicBlock[] = [];
        for (const sourceBlock of sourceBlocks) {
            const targetBlock = new BasicBlock();
            for (const stmt of sourceBlock.getStmts()) {
                targetBlock.addStmt(this.copyStmt(stmt)!);
            }
            sourceToTarget.set(sourceBlock, targetBlock);
            targetBlocks.push(targetBlock);
        }
        for (const sourceBlock of sourceBlocks) {
            const targetBlock = sourceToTarget.get(sourceBlock)!;
            for (const predecessor of sourceBlock.getPredecessors()) {
                const targetPredecessor = sourceToTarget.get(predecessor)!;
                targetBlock.addPredecessorBlock(targetPredecessor);
            }
            for (const successor of sourceBlock.getSuccessors()) {
                const targetSuccessor = sourceToTarget.get(successor)!;
                targetBlock.addSuccessorBlock(targetSuccessor);
            }
        }
        return targetBlocks;
    }

    private copyStmt(sourceStmt: Stmt): Stmt | null {
        if (sourceStmt instanceof ArkAssignStmt) {
            return new ArkAssignStmt(sourceStmt.getLeftOp(), sourceStmt.getRightOp());
        } else if (sourceStmt instanceof ArkInvokeStmt) {
            return new ArkInvokeStmt(sourceStmt.getInvokeExpr());
        } else if (sourceStmt instanceof ArkIfStmt) {
            return new ArkIfStmt(sourceStmt.getConditionExpr());
        } else if (sourceStmt instanceof ArkReturnStmt) {
            return new ArkReturnStmt(sourceStmt.getOp());
        } else if (sourceStmt instanceof ArkReturnVoidStmt) {
            return new ArkReturnVoidStmt();
        } else if (sourceStmt instanceof ArkThrowStmt) {
            return new ArkThrowStmt(sourceStmt.getOp());
        }
        return null;
    }

    private buildSwitch(blockBuilderToCfgBlock: Map<Block, BasicBlock>, blockBuildersContainSwitch: Block[],
                        valueAndStmtsOfSwitchAndCasesAll: ValueAndStmts[][], arkIRTransformer: ArkIRTransformer,
                        basicBlockSet: Set<BasicBlock>): void {
        for (let i = 0; i < blockBuildersContainSwitch.length; i++) {
            const blockBuilderContainSwitch = blockBuildersContainSwitch[i];

            if (!blockBuilderToCfgBlock.has(blockBuilderContainSwitch)) {
                logger.error(
                    `can't find basicBlock corresponding to the blockBuilder.`);
                continue;
            }

            const blockContainSwitch = blockBuilderToCfgBlock.get(blockBuilderContainSwitch)!;
            const valueAndStmtsOfSwitch = valueAndStmtsOfSwitchAndCasesAll[i][0];
            const stmtsOfSwitch = valueAndStmtsOfSwitch.stmts;
            stmtsOfSwitch.forEach((stmt: Stmt) => {
                blockContainSwitch.addStmt(stmt);
            });

            const stmtsCnt = blockBuilderContainSwitch.stmts.length;
            const switchStmtBuilder = blockBuilderContainSwitch.stmts[stmtsCnt - 1] as SwitchStatementBuilder;
            const cases = switchStmtBuilder.cases;
            let nonEmptyCaseCnt = 0;
            for (const currCase of cases) {
                if (currCase.stmt.block) { // there are stmts after this case
                    nonEmptyCaseCnt++;
                }
            }
            if (nonEmptyCaseCnt === 0) {
                continue;
            }

            const caseCnt = cases.length;
            const caseIfBlocks = this.generateIfBlocksForCases(valueAndStmtsOfSwitchAndCasesAll[i], caseCnt,
                blockContainSwitch, basicBlockSet, arkIRTransformer);
            this.linkIfBlockAndCaseBlock(blockContainSwitch, caseIfBlocks, switchStmtBuilder,
                blockBuilderToCfgBlock);
        }
    }

    private generateIfBlocksForCases(valueAndStmtsOfSwitchAndCases: ValueAndStmts[], caseCnt: number,
                                     blockContainSwitch: BasicBlock, basicBlockSet: Set<BasicBlock>,
                                     arkIRTransformer: ArkIRTransformer): BasicBlock[] {
        const valueAndStmtsOfSwitch = valueAndStmtsOfSwitchAndCases[0];
        const valueOfSwitch = valueAndStmtsOfSwitch.value;
        const caseIfBlocks: BasicBlock[] = [];

        for (let j = 0; j < caseCnt; j++) {
            let caseIfBlock: BasicBlock;
            if (j === 0) {
                caseIfBlock = blockContainSwitch;
            } else {
                caseIfBlock = new BasicBlock();
                basicBlockSet.add(caseIfBlock);
            }
            caseIfBlocks.push(caseIfBlock);

            const caseValueAndStmts = valueAndStmtsOfSwitchAndCases[j + 1];
            const caseValue = caseValueAndStmts.value;
            const caseStmts = caseValueAndStmts.stmts;
            caseStmts.forEach((stmt: Stmt) => {
                caseIfBlock.addStmt(stmt);
            });
            const caseIfStmts = arkIRTransformer.generateIfStmtForValues(valueOfSwitch,
                valueAndStmtsOfSwitch.valueOriginalPositions, caseValue,
                caseValueAndStmts.valueOriginalPositions);
            caseIfStmts.forEach((stmt: Stmt) => {
                caseIfBlock.addStmt(stmt);
            });
        }
        return caseIfBlocks;
    }

    private linkIfBlockAndCaseBlock(blockContainSwitch: BasicBlock, caseIfBlocks: BasicBlock[],
                                    switchStmtBuilder: SwitchStatementBuilder,
                                    blockBuilderToCfgBlock: Map<Block, BasicBlock>): boolean {
        const successorsOfBlockContainSwitch = Array.from(blockContainSwitch.getSuccessors());
        const expectedSuccessorsOfCaseIfBlock: BasicBlock[] = [];
        const defaultStmtBuilder = switchStmtBuilder.default;
        if (defaultStmtBuilder && defaultStmtBuilder.block) {
            expectedSuccessorsOfCaseIfBlock.push(...successorsOfBlockContainSwitch.splice(-1, 1));
        } else {
            const afterSwitchStmtBuilder = switchStmtBuilder.afterSwitch;
            const afterSwitchBlockBuilder = afterSwitchStmtBuilder?.block;
            if (!afterSwitchBlockBuilder || !blockBuilderToCfgBlock.has(afterSwitchBlockBuilder)) {
                logger.error(`can't find basicBlock corresponding to the blockBuilder.`);
                return false;
            }
            expectedSuccessorsOfCaseIfBlock.push(blockBuilderToCfgBlock.get(afterSwitchBlockBuilder)!);
        }
        const caseCnt = switchStmtBuilder.cases.length;
        for (let i = caseCnt - 1; i >= 0; i--) {
            const currCase = switchStmtBuilder.cases[i];
            if (currCase.stmt.block) {
                expectedSuccessorsOfCaseIfBlock.push(...successorsOfBlockContainSwitch.splice(-1, 1));
            } else { // if there are no stmts after this case, reuse the successor of the next case
                expectedSuccessorsOfCaseIfBlock.push(...expectedSuccessorsOfCaseIfBlock.slice(-1));
            }
        }
        expectedSuccessorsOfCaseIfBlock.reverse();

        blockContainSwitch.getSuccessors().forEach((successor) => {
            successor.getPredecessors().splice(0, 1);
        });
        blockContainSwitch.getSuccessors().splice(0);
        for (let j = 0; j < caseCnt; j++) {
            const caseIfBlock = caseIfBlocks[j];
            caseIfBlock.addSuccessorBlock(expectedSuccessorsOfCaseIfBlock[j]);
            expectedSuccessorsOfCaseIfBlock[j].addPredecessorBlock(caseIfBlock);
            if (j === caseCnt - 1) { // the false branch of last case should be default or block after switch statement
                caseIfBlock.addSuccessorBlock(expectedSuccessorsOfCaseIfBlock[j + 1]);
                expectedSuccessorsOfCaseIfBlock[j + 1].addPredecessorBlock(caseIfBlock);
            } else {
                caseIfBlock.addSuccessorBlock(caseIfBlocks[j + 1]);
                caseIfBlocks[j + 1].addPredecessorBlock(caseIfBlock);
            }
        }
        return true;
    }
}