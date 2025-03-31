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

import {
    AbstractExpr,
    AbstractInvokeExpr,
    ArkCastExpr,
    ArkConditionExpr,
    ArkInstanceInvokeExpr,
    ArkStaticInvokeExpr,
    BinaryOperator,
    NormalBinaryOperator,
    RelationalBinaryOperator,
    UnaryOperator,
} from '../base/Expr';
import { ArkCaughtExceptionRef, ArkInstanceFieldRef, ArkParameterRef, ArkThisRef, GlobalRef } from '../base/Ref';
import { Value } from '../base/Value';
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
import { AliasType, AliasTypeDeclaration, BooleanType, ClassType, UnknownType } from '../base/Type';
import { ValueUtil } from './ValueUtil';
import {
    ClassSignature,
    FieldSignature,
    LocalSignature,
    MethodSignature,
    MethodSubSignature,
} from '../model/ArkSignature';

import { IRUtils } from './IRUtils';
import { ArkMethod } from '../model/ArkMethod';
import { buildArkMethodFromArkClass } from '../model/builder/ArkMethodBuilder';
import { ArkSignatureBuilder } from '../model/builder/ArkSignatureBuilder';
import {
    COMPONENT_BRANCH_FUNCTION,
    COMPONENT_CREATE_FUNCTION,
    COMPONENT_IF,
    COMPONENT_POP_FUNCTION,
    COMPONENT_REPEAT,
} from './EtsConst';
import { FullPosition, LineColPosition } from '../base/Position';
import { ModelUtils } from './ModelUtils';
import { Builtin } from './Builtin';
import { DEFAULT } from './TSConst';
import { buildModifiers } from '../model/builder/builderUtils';
import { ArkValueTransformer } from './ArkValueTransformer';

export type ValueAndStmts = {
    value: Value,
    valueOriginalPositions: FullPosition[], // original positions of value and its uses
    stmts: Stmt[]
};

export class DummyStmt extends Stmt {
    constructor(text: string) {
        super();
        this.text = text;
    }

    public toString(): string {
        return this.text!;
    }
}

export class ArkIRTransformer {
    public static readonly DUMMY_LOOP_INITIALIZER_STMT = 'LoopInitializer';
    public static readonly DUMMY_CONDITIONAL_OPERATOR = 'ConditionalOperator';
    public static readonly DUMMY_CONDITIONAL_OPERATOR_IF_TRUE_STMT = ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR + 'IfTrue';
    public static readonly DUMMY_CONDITIONAL_OPERATOR_IF_FALSE_STMT = ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR + 'IfFalse';
    public static readonly DUMMY_CONDITIONAL_OPERATOR_END_STMT = ArkIRTransformer.DUMMY_CONDITIONAL_OPERATOR + 'End';

    private sourceFile: ts.SourceFile;
    private declaringMethod: ArkMethod;
    private inBuilderMethod = false;
    private builderMethodContextFlag = false;
    private stmtsHaveOriginalText: Set<Stmt> = new Set();
    private arkValueTransformer: ArkValueTransformer;

    constructor(sourceFile: ts.SourceFile, declaringMethod: ArkMethod) {
        this.sourceFile = sourceFile;
        this.declaringMethod = declaringMethod;
        this.inBuilderMethod = ModelUtils.isArkUIBuilderMethod(declaringMethod);
        this.arkValueTransformer = new ArkValueTransformer(this, sourceFile, this.declaringMethod);
    }

    public getLocals(): Set<Local> {
        return this.arkValueTransformer.getLocals();
    }

    public getGlobals(): Map<string, GlobalRef> | null {
        return this.arkValueTransformer.getGlobals();
    }

    public getThisLocal(): Local {
        return this.arkValueTransformer.getThisLocal();
    }

    public getAliasTypeMap(): Map<string, [AliasType, AliasTypeDeclaration]> {
        return this.arkValueTransformer.getAliasTypeMap();
    }

    public prebuildStmts(): Stmt[] {
        const stmts: Stmt[] = [];
        let index = 0;
        for (const methodParameter of this.declaringMethod.getParameters()) {
            const parameterRef = new ArkParameterRef(index, methodParameter.getType());
            stmts.push(new ArkAssignStmt(
                this.arkValueTransformer.addNewLocal(methodParameter.getName(), parameterRef.getType()),
                parameterRef));
            index++;
        }

        const thisRef = new ArkThisRef(this.arkValueTransformer.getThisLocal().getType() as ClassType);
        stmts.push(new ArkAssignStmt(this.arkValueTransformer.getThisLocal(), thisRef));
        return stmts;
    }

    public tsNodeToStmts(node: ts.Node): Stmt[] {
        let stmts: Stmt[] = [];
        if (ts.isExpressionStatement(node)) {
            stmts = this.expressionStatementToStmts(node);
        } else if (ts.isTypeAliasDeclaration(node)) {
            stmts = this.typeAliasDeclarationToStmts(node);
        } else if (ts.isBlock(node)) {
            stmts = this.blockToStmts(node);
        } else if (ts.isForStatement(node)) {
            stmts = this.forStatementToStmts(node);
        } else if (ts.isForInStatement(node) || ts.isForOfStatement(node)) {
            stmts = this.rangeForStatementToStmts(node);
        } else if (ts.isWhileStatement(node)) {
            stmts = this.whileStatementToStmts(node);
        } else if (ts.isDoStatement(node)) {
            stmts = this.doStatementToStmts(node);
        } else if (ts.isVariableStatement(node)) {
            stmts = this.variableStatementToStmts(node);
        } else if (ts.isVariableDeclarationList(node)) {
            stmts = this.variableDeclarationListToStmts(node);
        } else if (ts.isIfStatement(node)) {
            stmts = this.ifStatementToStmts(node);
        } else if (ts.isBreakStatement(node) || ts.isContinueStatement(node)) {
            stmts = this.gotoStatementToStmts(node);
        } else if (ts.isThrowStatement(node)) {
            stmts = this.throwStatementToStmts(node);
        } else if (ts.isCatchClause(node)) {
            stmts = this.catchClauseToStmts(node);
        } else if (ts.isReturnStatement(node)) {
            stmts = this.returnStatementToStmts(node);
        } else if (ts.isFunctionDeclaration(node)) {
            stmts = this.functionDeclarationToStmts(node);
        } else if (ts.isExportAssignment(node)) {
            stmts = this.expressionInExportToStmts(node.expression);
        }

        this.mapStmtsToTsStmt(stmts, node);
        if (stmts.length > 0) {
            IRUtils.setLeadingComments(stmts[0], node, this.sourceFile, this.declaringMethod.getDeclaringArkFile().getScene().getOptions());
        }
        return stmts;
    }

    public tsNodeToValueAndStmts(node: ts.Node): ValueAndStmts {
        return this.arkValueTransformer.tsNodeToValueAndStmts(node);
    }

    private functionDeclarationToStmts(functionDeclarationNode: ts.FunctionDeclaration): Stmt[] {
        const declaringClass = this.declaringMethod.getDeclaringArkClass();
        const arkMethod = new ArkMethod();
        if (this.builderMethodContextFlag) {
            ModelUtils.implicitArkUIBuilderMethods.add(arkMethod);
        }
        buildArkMethodFromArkClass(functionDeclarationNode, declaringClass, arkMethod, this.sourceFile, this.declaringMethod);
        return [];
    }

    private returnStatementToStmts(returnStatement: ts.ReturnStatement): Stmt[] {
        const stmts: Stmt[] = [];
        if (returnStatement.expression) {
            let {
                value: exprValue,
                valueOriginalPositions: exprPositions,
                stmts: exprStmts,
            } = this.tsNodeToValueAndStmts(returnStatement.expression);
            stmts.push(...exprStmts);
            if (IRUtils.moreThanOneAddress(exprValue)) {
                ({
                    value: exprValue,
                    valueOriginalPositions: exprPositions,
                    stmts: exprStmts,
                } = this.generateAssignStmtForValue(exprValue, exprPositions));
                stmts.push(...exprStmts);
            }
            const returnStmt = new ArkReturnStmt(exprValue);
            returnStmt.setOperandOriginalPositions(exprPositions);
            stmts.push(returnStmt);
        } else {
            stmts.push(new ArkReturnVoidStmt());
        }
        return stmts;
    }

    private blockToStmts(block: ts.Block): Stmt[] {
        const stmts: Stmt[] = [];
        for (const statement of block.statements) {
            stmts.push(...this.tsNodeToStmts(statement));
        }
        return stmts;
    }

    private expressionStatementToStmts(expressionStatement: ts.ExpressionStatement): Stmt[] {
        return this.expressionToStmts(expressionStatement.expression);
    }

    public expressionToStmts(expression: ts.Expression): Stmt[] {
        const {
            value: exprValue,
            valueOriginalPositions: exprPositions,
            stmts: stmts,
        } = this.tsNodeToValueAndStmts(expression);
        if (exprValue instanceof AbstractInvokeExpr) {
            const invokeStmt = new ArkInvokeStmt(exprValue);
            invokeStmt.setOperandOriginalPositions(exprPositions);
            stmts.push(invokeStmt);

            let hasRepeat: boolean = false;
            for (const stmt of stmts) {
                if ((stmt instanceof ArkAssignStmt) && (stmt.getRightOp() instanceof ArkStaticInvokeExpr)) {
                    const rightOp = stmt.getRightOp() as ArkStaticInvokeExpr;
                    if (rightOp.getMethodSignature().getMethodSubSignature().getMethodName() === COMPONENT_REPEAT) {
                        const createMethodSignature = ArkSignatureBuilder.buildMethodSignatureFromClassNameAndMethodName(COMPONENT_REPEAT, COMPONENT_CREATE_FUNCTION);
                        const createInvokeExpr = new ArkStaticInvokeExpr(createMethodSignature, rightOp.getArgs());
                        stmt.setRightOp(createInvokeExpr);
                        hasRepeat = true;
                    }
                }
            }
            if (hasRepeat) {
                const popMethodSignature = ArkSignatureBuilder.buildMethodSignatureFromClassNameAndMethodName(COMPONENT_REPEAT, COMPONENT_POP_FUNCTION);
                const popInvokeExpr = new ArkStaticInvokeExpr(popMethodSignature, []);
                const popInvokeStmt = new ArkInvokeStmt(popInvokeExpr);
                stmts.push(popInvokeStmt);
            }
        } else if (exprValue instanceof AbstractExpr) {
            const { stmts: exprStmts } = this.generateAssignStmtForValue(exprValue, exprPositions);
            stmts.push(...exprStmts);
        }
        return stmts;
    }

    private typeAliasDeclarationToStmts(typeAliasDeclaration: ts.TypeAliasDeclaration): Stmt[] {
        const aliasName = typeAliasDeclaration.name.text;
        const originalType = this.arkValueTransformer.resolveTypeNode(typeAliasDeclaration.type);
        const aliasType = new AliasType(aliasName, originalType,
            new LocalSignature(aliasName, this.declaringMethod.getSignature()));
        const modifiers = typeAliasDeclaration.modifiers ? buildModifiers(typeAliasDeclaration) : 0;
        aliasType.setModifiers(modifiers);
        const sourceCode = typeAliasDeclaration.getText(this.sourceFile);
        const aliasTypePosition = LineColPosition.buildFromNode(typeAliasDeclaration, this.sourceFile);
        const aliasTypeDeclaration = new AliasTypeDeclaration(sourceCode, aliasTypePosition)
        this.arkValueTransformer.getAliasTypeMap().set(aliasName, [aliasType, aliasTypeDeclaration]);
        return [];
    }

    public switchStatementToValueAndStmts(switchStatement: ts.SwitchStatement): ValueAndStmts[] {
        const valueAndStmtsOfSwitchAndCases: ValueAndStmts[] = [];
        const exprStmts: Stmt[] = [];
        let {
            value: exprValue,
            valueOriginalPositions: exprPositions,
            stmts: exprTempStmts,
        } = this.tsNodeToValueAndStmts(switchStatement.expression);
        exprStmts.push(...exprTempStmts);
        if (IRUtils.moreThanOneAddress(exprValue)) {
            ({ value: exprValue, valueOriginalPositions: exprPositions, stmts: exprTempStmts } =
                this.generateAssignStmtForValue(exprValue, exprPositions));
            exprStmts.push(...exprTempStmts);
        }
        valueAndStmtsOfSwitchAndCases.push(
            { value: exprValue, valueOriginalPositions: exprPositions, stmts: exprStmts });

        for (const clause of switchStatement.caseBlock.clauses) {
            if (ts.isCaseClause(clause)) {
                const clauseStmts: Stmt[] = [];
                let {
                    value: clauseValue,
                    valueOriginalPositions: clausePositions,
                    stmts: clauseTempStmts,
                } = this.tsNodeToValueAndStmts(clause.expression);
                clauseStmts.push(...clauseTempStmts);
                if (IRUtils.moreThanOneAddress(clauseValue)) {
                    ({
                        value: clauseValue,
                        valueOriginalPositions: clausePositions,
                        stmts: clauseTempStmts,
                    } = this.generateAssignStmtForValue(clauseValue, clausePositions));
                    clauseStmts.push(...clauseTempStmts);
                }
                valueAndStmtsOfSwitchAndCases.push(
                    { value: clauseValue, valueOriginalPositions: clausePositions, stmts: clauseStmts });
            }
        }
        return valueAndStmtsOfSwitchAndCases;
    }

    private forStatementToStmts(forStatement: ts.ForStatement): Stmt[] {
        const stmts: Stmt[] = [];
        if (forStatement.initializer) {
            stmts.push(...this.tsNodeToValueAndStmts(forStatement.initializer).stmts);
        }
        const dummyInitializerStmt = new DummyStmt(ArkIRTransformer.DUMMY_LOOP_INITIALIZER_STMT);
        stmts.push(dummyInitializerStmt);

        if (forStatement.condition) {
            const {
                value: conditionValue,
                stmts: conditionStmts,
            } = this.arkValueTransformer.conditionToValueAndStmts(forStatement.condition);
            stmts.push(...conditionStmts);
            stmts.push(new ArkIfStmt(conditionValue as ArkConditionExpr));
        }
        if (forStatement.incrementor) {
            stmts.push(...this.tsNodeToValueAndStmts(forStatement.incrementor).stmts);
        }
        return stmts;
    }

    private rangeForStatementToStmts(forOfStatement: ts.ForOfStatement | ts.ForInStatement): Stmt[] {
        const stmts: Stmt[] = [];
        let {
            value: iterableValue,
            valueOriginalPositions: iterablePositions,
            stmts: iterableStmts,
        } = this.tsNodeToValueAndStmts(forOfStatement.expression);
        stmts.push(...iterableStmts);
        if (!(iterableValue instanceof Local)) {
            ({ value: iterableValue, valueOriginalPositions: iterablePositions, stmts: iterableStmts } =
                this.generateAssignStmtForValue(iterableValue, iterablePositions));
            stmts.push(...iterableStmts);
        }
        const iteratorMethodSubSignature = new MethodSubSignature(Builtin.ITERATOR_FUNCTION, [],
            Builtin.ITERATOR_CLASS_TYPE);
        const iteratorMethodSignature = new MethodSignature(ClassSignature.DEFAULT, iteratorMethodSubSignature);
        const iteratorInvokeExpr = new ArkInstanceInvokeExpr(iterableValue as Local, iteratorMethodSignature, []);
        const iteratorInvokeExprPositions = [iterablePositions[0], ...iterablePositions];
        const {
            value: iterator,
            valueOriginalPositions: iteratorPositions,
            stmts: iteratorStmts,
        } = this.generateAssignStmtForValue(iteratorInvokeExpr, iteratorInvokeExprPositions);
        stmts.push(...iteratorStmts);
        (iterator as Local).setType(Builtin.ITERATOR_CLASS_TYPE);

        const nextMethodSubSignature = new MethodSubSignature(Builtin.ITERATOR_NEXT, [],
            Builtin.ITERATOR_RESULT_CLASS_TYPE);
        const nextMethodSignature = new MethodSignature(ClassSignature.DEFAULT, nextMethodSubSignature);
        const iteratorNextInvokeExpr = new ArkInstanceInvokeExpr(iterator as Local, nextMethodSignature, []);
        const iteratorNextInvokeExprPositions = [iteratorPositions[0], ...iteratorPositions];
        const {
            value: iteratorResult,
            valueOriginalPositions: iteratorResultPositions,
            stmts: iteratorResultStmts,
        } = this.generateAssignStmtForValue(iteratorNextInvokeExpr,
            iteratorNextInvokeExprPositions);
        stmts.push(...iteratorResultStmts);
        (iteratorResult as Local).setType(Builtin.ITERATOR_RESULT_CLASS_TYPE);
        const doneFieldSignature = new FieldSignature(Builtin.ITERATOR_RESULT_DONE,
            Builtin.ITERATOR_RESULT_CLASS_SIGNATURE, BooleanType.getInstance(), false);
        const doneFieldRef = new ArkInstanceFieldRef(iteratorResult as Local, doneFieldSignature);
        const doneFieldRefPositions = [iteratorResultPositions[0], ...iteratorResultPositions];
        const {
            value: doneFlag,
            valueOriginalPositions: doneFlagPositions,
            stmts: doneFlagStmts,
        } = this.generateAssignStmtForValue(doneFieldRef, doneFieldRefPositions);
        stmts.push(...doneFlagStmts);
        (doneFlag as Local).setType(BooleanType.getInstance());
        const conditionExpr = new ArkConditionExpr(doneFlag, ValueUtil.getBooleanConstant(true), RelationalBinaryOperator.Equality);
        const conditionExprPositions = [doneFlagPositions[0], ...doneFlagPositions, FullPosition.DEFAULT];
        const ifStmt = new ArkIfStmt(conditionExpr);
        ifStmt.setOperandOriginalPositions(conditionExprPositions);
        stmts.push(ifStmt);

        const valueFieldSignature = new FieldSignature(Builtin.ITERATOR_RESULT_VALUE,
            Builtin.ITERATOR_RESULT_CLASS_SIGNATURE, UnknownType.getInstance(), false);
        const valueFieldRef = new ArkInstanceFieldRef(iteratorResult as Local, valueFieldSignature);
        const valueFieldRefPositions = [iteratorResultPositions[0], ...iteratorResultPositions];
        const {
            value: yieldValue,
            valueOriginalPositions: yieldValuePositions,
            stmts: yieldValueStmts,
        } = this.generateAssignStmtForValue(valueFieldRef, valueFieldRefPositions);
        stmts.push(...yieldValueStmts);

        const castExpr = new ArkCastExpr(yieldValue, UnknownType.getInstance());
        const castExprPositions = [yieldValuePositions[0], ...yieldValuePositions];
        const initializerNode = forOfStatement.initializer;
        if (ts.isVariableDeclarationList(initializerNode)) {
            const isConst = (initializerNode.flags & ts.NodeFlags.Const) !== 0;
            const {
                value: initValue, valueOriginalPositions: initOriPos, stmts: initStmts,
            } = this.arkValueTransformer.variableDeclarationToValueAndStmts(initializerNode.declarations[0], isConst,
                false);
            const assignStmt = new ArkAssignStmt(initValue, castExpr);
            assignStmt.setOperandOriginalPositions([...initOriPos, ...castExprPositions]);
            stmts.push(assignStmt, ...initStmts);
        } else { // initializer maybe an expression
            const {
                value: initValue, valueOriginalPositions: initOriPos, stmts: initStmts,
            } = this.tsNodeToValueAndStmts(initializerNode);
            const assignStmt = new ArkAssignStmt(initValue, castExpr);
            assignStmt.setOperandOriginalPositions([...initOriPos, ...castExprPositions]);
            stmts.push(...initStmts, assignStmt);
        }
        return stmts;
    }

    private whileStatementToStmts(whileStatement: ts.WhileStatement): Stmt[] {
        const stmts: Stmt[] = [];
        const dummyInitializerStmt = new DummyStmt(ArkIRTransformer.DUMMY_LOOP_INITIALIZER_STMT);
        stmts.push(dummyInitializerStmt);

        const {
            value: conditionExpr,
            stmts: conditionStmts,
        } = this.arkValueTransformer.conditionToValueAndStmts(whileStatement.expression);
        stmts.push(...conditionStmts);
        stmts.push(new ArkIfStmt(conditionExpr as ArkConditionExpr));
        return stmts;
    }

    private doStatementToStmts(doStatement: ts.DoStatement): Stmt[] {
        const stmts: Stmt[] = [];
        const {
            value: conditionExpr,
            stmts: conditionStmts,
        } = this.arkValueTransformer.conditionToValueAndStmts(doStatement.expression);
        stmts.push(...conditionStmts);
        stmts.push(new ArkIfStmt(conditionExpr as ArkConditionExpr));
        return stmts;
    }

    private variableStatementToStmts(variableStatement: ts.VariableStatement): Stmt[] {
        return this.variableDeclarationListToStmts(variableStatement.declarationList);
    }

    private variableDeclarationListToStmts(variableDeclarationList: ts.VariableDeclarationList): Stmt[] {
        return this.arkValueTransformer.variableDeclarationListToValueAndStmts(variableDeclarationList).stmts;
    }

    private ifStatementToStmts(ifStatement: ts.IfStatement): Stmt[] {
        const stmts: Stmt[] = [];
        if (this.inBuilderMethod) {
            const {
                value: conditionExpr,
                valueOriginalPositions: conditionExprPositions,
                stmts: conditionStmts,
            } = this.arkValueTransformer.conditionToValueAndStmts(ifStatement.expression);
            stmts.push(...conditionStmts);
            const createMethodSignature = ArkSignatureBuilder.buildMethodSignatureFromClassNameAndMethodName(COMPONENT_IF, COMPONENT_CREATE_FUNCTION);
            const {
                value: conditionLocal,
                valueOriginalPositions: conditionLocalPositions,
                stmts: assignConditionStmts
            } = this.generateAssignStmtForValue(conditionExpr, conditionExprPositions);
            stmts.push(...assignConditionStmts);
            const createInvokeExpr = new ArkStaticInvokeExpr(createMethodSignature, [conditionLocal]);
            const createInvokeExprPositions = [conditionLocalPositions[0], ...conditionLocalPositions];
            const { stmts: createStmts } = this.generateAssignStmtForValue(createInvokeExpr,
                createInvokeExprPositions);
            stmts.push(...createStmts);
            const branchMethodSignature = ArkSignatureBuilder.buildMethodSignatureFromClassNameAndMethodName(COMPONENT_IF, COMPONENT_BRANCH_FUNCTION);
            const branchInvokeExpr = new ArkStaticInvokeExpr(branchMethodSignature, [ValueUtil.getOrCreateNumberConst(0)]);
            const branchInvokeExprPositions = [conditionLocalPositions[0], FullPosition.DEFAULT];
            const branchInvokeStmt = new ArkInvokeStmt(branchInvokeExpr);
            branchInvokeStmt.setOperandOriginalPositions(branchInvokeExprPositions);
            stmts.push(branchInvokeStmt);
            stmts.push(...this.tsNodeToStmts(ifStatement.thenStatement));
            if (ifStatement.elseStatement) {
                const branchElseMethodSignature = ArkSignatureBuilder.buildMethodSignatureFromClassNameAndMethodName(COMPONENT_IF, COMPONENT_BRANCH_FUNCTION);
                const branchElseInvokeExpr = new ArkStaticInvokeExpr(branchElseMethodSignature, [ValueUtil.getOrCreateNumberConst(1)]);
                const branchElseInvokeExprPositions = [FullPosition.buildFromNode(ifStatement.elseStatement,
                    this.sourceFile), FullPosition.DEFAULT];
                const branchElseInvokeStmt = new ArkInvokeStmt(branchElseInvokeExpr);
                branchElseInvokeStmt.setOperandOriginalPositions(branchElseInvokeExprPositions);
                stmts.push(branchElseInvokeStmt);

                stmts.push(...this.tsNodeToStmts(ifStatement.elseStatement));
            }
            const popMethodSignature = ArkSignatureBuilder.buildMethodSignatureFromClassNameAndMethodName(COMPONENT_IF, COMPONENT_POP_FUNCTION);
            const popInvokeExpr = new ArkStaticInvokeExpr(popMethodSignature, []);
            const popInvokeStmt = new ArkInvokeStmt(popInvokeExpr);
            stmts.push(popInvokeStmt);
        } else {
            const {
                value: conditionExpr,
                valueOriginalPositions: conditionExprPositions,
                stmts: conditionStmts,
            } = this.arkValueTransformer.conditionToValueAndStmts(ifStatement.expression);
            stmts.push(...conditionStmts);
            const ifStmt = new ArkIfStmt(conditionExpr as ArkConditionExpr);
            ifStmt.setOperandOriginalPositions(conditionExprPositions);
            stmts.push(ifStmt);
        }
        return stmts;
    }

    private gotoStatementToStmts(gotoStatement: ts.BreakStatement | ts.ContinueStatement): Stmt[] {
        return [];
    }

    private throwStatementToStmts(throwStatement: ts.ThrowStatement): Stmt[] {
        const stmts: Stmt[] = [];
        const {
            value: throwValue,
            valueOriginalPositions: throwValuePositions,
            stmts: throwStmts,
        } = this.tsNodeToValueAndStmts(throwStatement.expression);
        stmts.push(...throwStmts);
        const throwStmt = new ArkThrowStmt(throwValue);
        throwStmt.setOperandOriginalPositions(throwValuePositions);
        stmts.push(throwStmt);
        return stmts;
    }

    private catchClauseToStmts(catchClause: ts.CatchClause): Stmt[] {
        const stmts: Stmt[] = [];
        if (catchClause.variableDeclaration) {
            const {
                value: catchValue, valueOriginalPositions: catchOriPos, stmts: catchStmts,
            } = this.arkValueTransformer.variableDeclarationToValueAndStmts(catchClause.variableDeclaration, false,
                false);
            const caughtExceptionRef = new ArkCaughtExceptionRef(UnknownType.getInstance());
            const assignStmt = new ArkAssignStmt(catchValue, caughtExceptionRef);
            assignStmt.setOperandOriginalPositions(catchOriPos);
            stmts.push(assignStmt, ...catchStmts);
        }
        return stmts;
    }

    private expressionInExportToStmts(expression: ts.Node): Stmt[] {
        if (ts.isNewExpression(expression) || ts.isObjectLiteralExpression(expression)) {
            return this.newClassInExportToStmts(expression);
        }
        return [];
    }

    private newClassInExportToStmts(expression: ts.NewExpression | ts.ObjectLiteralExpression): Stmt[] {
        let stmts: Stmt[] = [];
        let {
            value: rightValue,
            valueOriginalPositions: rightPositions,
            stmts: rightStmts,
        } = this.tsNodeToValueAndStmts(expression);
        stmts.push(...rightStmts);
        let leftValue = this.arkValueTransformer.addNewLocal(DEFAULT);
        let leftPositions = rightPositions;
        const assignStmt = new ArkAssignStmt(leftValue, rightValue);
        assignStmt.setOperandOriginalPositions([...leftPositions, ...rightPositions]);
        stmts.push(assignStmt);
        return stmts;
    }

    public mapStmtsToTsStmt(stmts: Stmt[], node: ts.Node): void {
        for (const stmt of stmts) {
            if (!this.stmtsHaveOriginalText.has(stmt)) {
                this.stmtsHaveOriginalText.add(stmt);
                stmt.setOriginPositionInfo(LineColPosition.buildFromNode(node, this.sourceFile));
                stmt.setOriginalText(node.getText(this.sourceFile));
            }
        }
    }

    public static tokenToUnaryOperator(token: ts.SyntaxKind): UnaryOperator | null {
        switch (token) {
            case ts.SyntaxKind.MinusToken:
                return UnaryOperator.Neg;
            case ts.SyntaxKind.TildeToken:
                return UnaryOperator.BitwiseNot;
            case ts.SyntaxKind.ExclamationToken:
                return UnaryOperator.LogicalNot;
            default:
                ;
        }
        return null;
    }

    public static tokenToBinaryOperator(token: ts.SyntaxKind): BinaryOperator | null {
        switch (token) {
            case ts.SyntaxKind.QuestionQuestionToken:
                return NormalBinaryOperator.NullishCoalescing;
            case ts.SyntaxKind.AsteriskAsteriskToken:
                return NormalBinaryOperator.Exponentiation;
            case ts.SyntaxKind.SlashToken:
                return NormalBinaryOperator.Division;
            case ts.SyntaxKind.PlusToken:
                return NormalBinaryOperator.Addition;
            case ts.SyntaxKind.MinusToken:
                return NormalBinaryOperator.Subtraction;
            case ts.SyntaxKind.AsteriskToken:
                return NormalBinaryOperator.Multiplication;
            case ts.SyntaxKind.PercentToken:
                return NormalBinaryOperator.Remainder;
            case ts.SyntaxKind.LessThanLessThanToken:
                return NormalBinaryOperator.LeftShift;
            case ts.SyntaxKind.GreaterThanGreaterThanToken:
                return NormalBinaryOperator.RightShift;
            case ts.SyntaxKind.GreaterThanGreaterThanGreaterThanToken:
                return NormalBinaryOperator.UnsignedRightShift;
            case ts.SyntaxKind.AmpersandToken:
                return NormalBinaryOperator.BitwiseAnd;
            case ts.SyntaxKind.BarToken:
                return NormalBinaryOperator.BitwiseOr;
            case ts.SyntaxKind.CaretToken:
                return NormalBinaryOperator.BitwiseXor;
            case ts.SyntaxKind.AmpersandAmpersandToken:
                return NormalBinaryOperator.LogicalAnd;
            case ts.SyntaxKind.BarBarToken:
                return NormalBinaryOperator.LogicalOr;
            case ts.SyntaxKind.LessThanToken:
                return RelationalBinaryOperator.LessThan;
            case ts.SyntaxKind.LessThanEqualsToken:
                return RelationalBinaryOperator.LessThanOrEqual;
            case ts.SyntaxKind.GreaterThanToken:
                return RelationalBinaryOperator.GreaterThan;
            case ts.SyntaxKind.GreaterThanEqualsToken:
                return RelationalBinaryOperator.GreaterThanOrEqual;
            case ts.SyntaxKind.EqualsEqualsToken:
                return RelationalBinaryOperator.Equality;
            case ts.SyntaxKind.ExclamationEqualsToken:
                return RelationalBinaryOperator.InEquality;
            case ts.SyntaxKind.EqualsEqualsEqualsToken:
                return RelationalBinaryOperator.StrictEquality;
            case ts.SyntaxKind.ExclamationEqualsEqualsToken:
                return RelationalBinaryOperator.StrictInequality;
        }
        return null;
    }

    public generateAssignStmtForValue(value: Value, valueOriginalPositions: FullPosition[]): ValueAndStmts {
        const leftOp = this.arkValueTransformer.generateTempLocal(value.getType());
        const leftOpPosition = valueOriginalPositions[0];
        const assignStmt = new ArkAssignStmt(leftOp, value);
        assignStmt.setOperandOriginalPositions([leftOpPosition, ...valueOriginalPositions]);
        return { value: leftOp, valueOriginalPositions: [leftOpPosition], stmts: [assignStmt] };
    }

    public generateIfStmtForValues(leftValue: Value, leftOpOriginalPositions: FullPosition[], rightValue: Value,
                                   rightOpOriginalPositions: FullPosition[]): Stmt[] {
        const stmts: Stmt[] = [];
        if (IRUtils.moreThanOneAddress(leftValue)) {
            const {
                value: tempLeftValue,
                valueOriginalPositions: tempLeftPositions,
                stmts: leftStmts,
            } = this.generateAssignStmtForValue(leftValue, leftOpOriginalPositions);
            stmts.push(...leftStmts);
            leftValue = tempLeftValue;
            leftOpOriginalPositions = tempLeftPositions;
        }
        if (IRUtils.moreThanOneAddress(rightValue)) {
            const {
                value: tempRightValue,
                valueOriginalPositions: tempRightPositions,
                stmts: rightStmts,
            } = this.generateAssignStmtForValue(rightValue, rightOpOriginalPositions);
            stmts.push(...rightStmts);
            rightValue = tempRightValue;
            rightOpOriginalPositions = tempRightPositions;
        }

        const conditionExpr = new ArkConditionExpr(leftValue, rightValue, RelationalBinaryOperator.Equality);
        const conditionPositions = [...leftOpOriginalPositions, ...rightOpOriginalPositions];
        const ifStmt = new ArkIfStmt(conditionExpr);
        ifStmt.setOperandOriginalPositions([...conditionPositions]);
        stmts.push(ifStmt);
        return stmts;
    }

    public setBuilderMethodContextFlag(builderMethodContextFlag: boolean): void {
        this.builderMethodContextFlag = builderMethodContextFlag;
    }
}