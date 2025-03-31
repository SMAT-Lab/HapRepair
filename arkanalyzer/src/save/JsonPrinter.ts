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

import { Printer } from './Printer';
import { ArkFile } from '../core/model/ArkFile';
import { ArkMethod } from '../core/model/ArkMethod';
import { ArkNamespace } from '../core/model/ArkNamespace';
import { ArkClass } from '../core/model/ArkClass';
import { ArkField } from '../core/model/ArkField';
import {
    AliasType,
    AnnotationNamespaceType,
    AnnotationType,
    AnnotationTypeQueryType,
    AnyType,
    ArrayType,
    BooleanType,
    ClassType,
    FunctionType,
    GenericType,
    LiteralType,
    NeverType,
    NullType,
    NumberType,
    PrimitiveType,
    StringType,
    TupleType,
    Type,
    UnclearReferenceType,
    UndefinedType,
    UnionType,
    UnknownType,
    VoidType,
} from '../core/base/Type';
import { Value } from '../core/base/Value';
import {
    ArkAssignStmt,
    ArkIfStmt,
    ArkInvokeStmt,
    ArkReturnStmt,
    ArkReturnVoidStmt,
    ArkThrowStmt,
    Stmt,
} from '../core/base/Stmt';
import {
    AbstractBinopExpr,
    AbstractExpr,
    AbstractInvokeExpr,
    ArkAwaitExpr,
    ArkCastExpr,
    ArkConditionExpr,
    ArkDeleteExpr,
    ArkInstanceInvokeExpr,
    ArkInstanceOfExpr,
    ArkNewArrayExpr,
    ArkNewExpr,
    ArkNormalBinopExpr,
    ArkPhiExpr,
    ArkPtrInvokeExpr,
    ArkStaticInvokeExpr,
    ArkTypeOfExpr,
    ArkUnopExpr,
    ArkYieldExpr,
} from '../core/base/Expr';
import { Constant } from '../core/base/Constant';
import { MethodParameter } from '../core/model/builder/ArkMethodBuilder';
import { ImportInfo } from '../core/model/ArkImport';
import { ExportInfo } from '../core/model/ArkExport';
import {
    ClassSignature,
    FieldSignature,
    FileSignature,
    LocalSignature,
    MethodSignature,
    NamespaceSignature,
} from '../core/model/ArkSignature';
import { LineColPosition } from '../core/base/Position';
import { AbstractFieldRef, AbstractRef, ArkArrayRef, ArkInstanceFieldRef, ArkParameterRef, ArkStaticFieldRef, ArkThisRef } from '../core/base/Ref';
import { Local } from '../core/base/Local';
import { Cfg } from '../core/graph/Cfg';
import { BasicBlock } from '../core/graph/BasicBlock';
import { ArkBody } from '../core/model/ArkBody';
import { Decorator } from '../core/base/Decorator';
import util from 'util';

export class JsonPrinter extends Printer {
    constructor(private arkFile: ArkFile) {
        super();
    }

    public dump(): string {
        const jsonObject = this.serializeArkFile(this.arkFile);
        return JSON.stringify(jsonObject, null, 2);
    }

    private serializeArkFile(file: ArkFile): object {
        return {
            signature: this.serializeFileSignature(file.getFileSignature()),
            namespaces: file.getNamespaces().map((ns) => this.serializeNamespace(ns)),
            classes: file.getClasses().map((cls) => this.serializeClass(cls)),
            importInfos: file.getImportInfos().map((info) => this.serializeImportInfo(info)),
            exportInfos: file.getExportInfos().map((info) => this.serializeExportInfo(info)),
        };
    }

    private serializeNamespace(namespace: ArkNamespace): object {
        return {
            signature: this.serializeNamespaceSignature(namespace.getSignature()),
            classes: namespace.getClasses().map((cls) => this.serializeClass(cls)),
            namespaces: namespace.getNamespaces().map((ns) => this.serializeNamespace(ns)),
        };
    }

    private serializeClass(clazz: ArkClass): object {
        return {
            signature: this.serializeClassSignature(clazz.getSignature()),
            modifiers: clazz.getModifiers(),
            decorators: clazz.getDecorators().map((decorator) => this.serializeDecorator(decorator)),
            typeParameters: clazz.getGenericsTypes()?.map((type) => this.serializeType(type)),
            superClassName: clazz.getSuperClassName(),
            implementedInterfaceNames: clazz.getImplementedInterfaceNames(),
            fields: clazz.getFields().map((field) => this.serializeField(field)),
            methods: clazz.getMethods(true).map((method) => this.serializeMethod(method)),
        };
    }

    private serializeField(field: ArkField): object {
        return {
            signature: this.serializeFieldSignature(field.getSignature()),
            modifiers: field.getModifiers(),
            decorators: field.getDecorators().map((decorator) => this.serializeDecorator(decorator)),
            questionToken: field.getQuestionToken(),
            exclamationToken: field.getExclamationToken(),
        };
    }

    private serializeMethod(method: ArkMethod): object {
        let body = method.getBody();
        return {
            signature: this.serializeMethodSignature(method.getSignature()),
            modifiers: method.getModifiers(),
            decorators: method.getDecorators().map((decorator) => this.serializeDecorator(decorator)),
            typeParameters: method.getGenericTypes()?.map(type => this.serializeType(type)),
            body: body && this.serializeMethodBody(body),
        };
    }

    private serializeMethodBody(body: ArkBody): object {
        return {
            locals: Array.from(body.getLocals().values()).map((local) => this.serializeLocal(local)),
            cfg: this.serializeCfg(body.getCfg()),
        };
    }

    private serializeMethodParameter(parameter: MethodParameter): object {
        return {
            name: parameter.getName(),
            type: this.serializeType(parameter.getType()),
            isOptional: parameter.isOptional(),
        };
    }

    private serializeImportInfo(importInfo: ImportInfo): object {
        return {
            importClauseName: importInfo.getImportClauseName(),
            importType: importInfo.getImportType(),
            importFrom: importInfo.getFrom(),
            nameBeforeAs: importInfo.getNameBeforeAs(),
            modifiers: importInfo.getModifiers(),
            originTsPosition: this.serializeLineColPosition(importInfo.getOriginTsPosition()),
        };
    }

    private serializeExportInfo(exportInfo: ExportInfo): object {
        return {
            exportClauseName: exportInfo.getExportClauseName(),
            exportClauseType: exportInfo.getExportClauseType(),
            exportFrom: exportInfo.getFrom(),
            nameBeforeAs: exportInfo.getNameBeforeAs(),
            isDefault: exportInfo.isDefault(),
            modifiers: exportInfo.getModifiers(),
            originTsPosition: this.serializeLineColPosition(exportInfo.getOriginTsPosition()),
        };
    }

    private serializeDecorator(decorator: Decorator): object {
        return {
            kind: decorator.getKind(),
        };
    }

    private serializeLineColPosition(position: LineColPosition): object {
        return {
            line: position.getLineNo(),
            col: position.getColNo(),
        };
    }

    private serializeType(type: Type): object {
        if (type === undefined) {
            throw new Error('Type is undefined');
        }

        if (type instanceof AnyType) {
            return {
                _: 'AnyType',
            };
        } else if (type instanceof UnknownType) {
            return {
                _: 'UnknownType',
            };
        } else if (type instanceof VoidType) {
            return {
                _: 'VoidType',
            };
        } else if (type instanceof NeverType) {
            return {
                _: 'NeverType',
            };
        } else if (type instanceof UnionType) {
            return {
                _: 'UnionType',
                types: type.getTypes().map((type) => this.serializeType(type)),
            };
        } else if (type instanceof TupleType) {
            return {
                _: 'TupleType',
                types: type.getTypes().map((type) => this.serializeType(type)),
            };
        } else if (type instanceof BooleanType) {
            return {
                _: 'BooleanType',
            };
        } else if (type instanceof NumberType) {
            return {
                _: 'NumberType',
            };
        } else if (type instanceof StringType) {
            return {
                _: 'StringType',
            };
        } else if (type instanceof NullType) {
            return {
                _: 'NullType',
            };
        } else if (type instanceof UndefinedType) {
            return {
                _: 'UndefinedType',
            };
        } else if (type instanceof LiteralType) {
            return {
                _: 'LiteralType',
                literal: type.getLiteralName(),
            };
        } else if (type instanceof PrimitiveType) {
            throw new Error('Unhandled PrimitiveType: ' + util.inspect(type, { showHidden: true, depth: null }));
        } else if (type instanceof ClassType) {
            return {
                _: 'ClassType',
                signature: this.serializeClassSignature(type.getClassSignature()),
                typeParameters: type.getRealGenericTypes()?.map((type) => this.serializeType(type)),
            };
        } else if (type instanceof FunctionType) {
            return {
                _: 'FunctionType',
                signature: this.serializeMethodSignature(type.getMethodSignature()),
                typeParameters: type.getRealGenericTypes()?.map((type) => this.serializeType(type)),
            };
        } else if (type instanceof ArrayType) {
            return {
                _: 'ArrayType',
                elementType: this.serializeType(type.getBaseType()),
                dimensions: type.getDimension(),
            };
        } else if (type instanceof UnclearReferenceType) {
            return {
                _: 'UnclearReferenceType',
                name: type.getName(),
                typeParameters: type.getGenericTypes().map((type) => this.serializeType(type)),
            };
        } else if (type instanceof GenericType) {
            let defaultType = type.getDefaultType();
            let constraint = type.getConstraint();
            return {
                _: 'GenericType',
                name: type.getName(),
                defaultType: defaultType && this.serializeType(defaultType),
                constraint: constraint && this.serializeType(constraint),
            };
        } else if (type instanceof AliasType) {
            return {
                _: 'AliasType',
                name: type.getName(),
                originalType: this.serializeType(type.getOriginalType()),
                signature: this.serializeLocalSignature(type.getSignature()),
            };
        } else if (type instanceof AnnotationNamespaceType) {
            return {
                _: 'AnnotationNamespaceType',
                originType: type.getOriginType(),
                namespaceSignature: this.serializeNamespaceSignature(type.getNamespaceSignature()),
            };
        } else if (type instanceof AnnotationTypeQueryType) {
            return {
                _: 'AnnotationTypeQueryType',
                originType: type.getOriginType(),
            };
        } else if (type instanceof AnnotationType) {
            throw new Error('Unhandled AnnotationType: ' + util.inspect(type, { showHidden: true, depth: null }));
        } else {
            throw new Error('Unhandled Type: ' + util.inspect(type, { showHidden: true, depth: null }));
        }
    }

    private serializeFileSignature(file: FileSignature): object {
        return {
            projectName: file.getProjectName(),
            fileName: file.getFileName(),
        };
    }

    private serializeNamespaceSignature(namespace: NamespaceSignature): object {
        let dns = namespace.getDeclaringNamespaceSignature();
        return {
            name: namespace.getNamespaceName(),
            declaringFile: this.serializeFileSignature(namespace.getDeclaringFileSignature()),
            declaringNamespace: dns && this.serializeNamespaceSignature(dns),
        };
    }

    private serializeClassSignature(clazz: ClassSignature): object {
        let dns = clazz.getDeclaringNamespaceSignature();
        return {
            name: clazz.getClassName(),
            declaringFile: this.serializeFileSignature(clazz.getDeclaringFileSignature()),
            declaringNamespace: dns && this.serializeNamespaceSignature(dns),
        };
    }

    private serializeFieldSignature(field: FieldSignature): object {
        let declaringSignature: ClassSignature | NamespaceSignature = field.getDeclaringSignature();
        let declaringClass;
        if (declaringSignature instanceof ClassSignature) {
            declaringClass = this.serializeClassSignature(declaringSignature);
        } else {
            declaringClass = this.serializeNamespaceSignature(declaringSignature);
        }
        return {
            declaringClass,
            name: field.getFieldName(),
            type: this.serializeType(field.getType()),
        };
    }

    private serializeMethodSignature(method: MethodSignature): object {
        return {
            declaringClass: this.serializeClassSignature(method.getDeclaringClassSignature()),
            name: method.getMethodSubSignature().getMethodName(),
            parameters: method
                .getMethodSubSignature()
                .getParameters()
                .map((param) => this.serializeMethodParameter(param)),
            returnType: this.serializeType(method.getType()),
        };
    }

    private serializeLocalSignature(signature: LocalSignature): object {
        return {
            name: signature.getName(),
            method: this.serializeMethodSignature(signature.getDeclaringMethodSubSignature()),
        };
    }

    private serializeCfg(cfg: Cfg): object {
        const visited = new Set<BasicBlock>();
        const stack: BasicBlock[] = [];
        const startingBlock = cfg.getStartingBlock();
        if (startingBlock) {
            stack.push(startingBlock);
        }
        let id = 0;
        while (stack.length > 0) {
            const block = stack.pop()!;
            if (visited.has(block)) {
                continue;
            }
            visited.add(block);
            block.setId(id++);
            stack.push(...block.getSuccessors());
        }
        return {
            blocks: Array.from(visited).map((block) => this.serializeBasicBlock(block)),
        };
    }

    private serializeBasicBlock(block: BasicBlock): object {
        const successors = block.getSuccessors().map((succ) => succ.getId());
        successors.sort((a, b) => a - b);
        const predecessors = block.getPredecessors().map((pred) => pred.getId());
        predecessors.sort((a, b) => a - b);
        return {
            id: block.getId(),
            successors,
            predecessors,
            stmts: block.getStmts().map((stmt) => this.serializeStmt(stmt)),
        };
    }

    private serializeLocal(local: Local): object {
        return {
            name: local.getName(),
            type: this.serializeType(local.getType()),
        };
    }

    private serializeValue(value: Value): object {
        if (value === undefined) {
            throw new Error('Value is undefined');
        }

        if (value instanceof Local) {
            return {
                _: 'Local',
                ...this.serializeLocal(value),
            };
        } else if (value instanceof Constant) {
            return {
                _: 'Constant',
                value: value.getValue(),
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkNewExpr) {
            return {
                _: 'NewExpr',
                classType: this.serializeType(value.getClassType()),
            };
        } else if (value instanceof ArkNewArrayExpr) {
            return {
                _: 'NewArrayExpr',
                elementType: this.serializeType(value.getBaseType()),
                size: this.serializeValue(value.getSize()),
            };
        } else if (value instanceof ArkDeleteExpr) {
            return {
                _: 'DeleteExpr',
                arg: this.serializeValue(value.getField()),
            };
        } else if (value instanceof ArkAwaitExpr) {
            return {
                _: 'AwaitExpr',
                arg: this.serializeValue(value.getPromise()),
            };
        } else if (value instanceof ArkYieldExpr) {
            return {
                _: 'YieldExpr',
                arg: this.serializeValue(value.getYieldValue()),
            };
        } else if (value instanceof ArkTypeOfExpr) {
            return {
                _: 'TypeOfExpr',
                arg: this.serializeValue(value.getOp()),
            };
        } else if (value instanceof ArkInstanceOfExpr) {
            return {
                _: 'InstanceOfExpr',
                arg: this.serializeValue(value.getOp()),
                checkType: this.serializeType(value.getCheckType()),
            };
        } else if (value instanceof ArkCastExpr) {
            return {
                _: 'CastExpr',
                arg: this.serializeValue(value.getOp()),
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkPhiExpr) {
            const args = value.getArgs();
            const argToBlock = value.getArgToBlock();
            return {
                _: 'PhiExpr',
                args: args.map((arg) => this.serializeValue(arg)),
                blocks: args.map((arg) => argToBlock.get(arg)!.getId()),
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkConditionExpr) {
            return {
                _: 'ConditionExpr',
                op: value.getOperator(),
                left: this.serializeValue(value.getOp1()),
                right: this.serializeValue(value.getOp2()),
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkNormalBinopExpr) {
            return {
                _: 'BinopExpr',
                op: value.getOperator(),
                left: this.serializeValue(value.getOp1()),
                right: this.serializeValue(value.getOp2()),
            };
        } else if (value instanceof AbstractBinopExpr) {
            return new Error('Unhandled BinopExpr: ' + util.inspect(value, { showHidden: true, depth: null }));
        } else if (value instanceof ArkUnopExpr) {
            return {
                _: 'UnopExpr',
                op: value.getOperator(),
                arg: this.serializeValue(value.getOp()),
            };
        } else if (value instanceof ArkInstanceInvokeExpr) {
            return {
                _: 'InstanceCallExpr',
                instance: this.serializeValue(value.getBase()),
                method: this.serializeMethodSignature(value.getMethodSignature()),
                args: value.getArgs().map((arg) => this.serializeValue(arg)),
            };
        } else if (value instanceof ArkStaticInvokeExpr) {
            return {
                _: 'StaticCallExpr',
                method: this.serializeMethodSignature(value.getMethodSignature()),
                args: value.getArgs().map((arg) => this.serializeValue(arg)),
            };
        } else if (value instanceof ArkPtrInvokeExpr) {
            return {
                _: 'PtrCallExpr',
                ptr: this.serializeValue(value.getFuncPtrLocal()),
                method: this.serializeMethodSignature(value.getMethodSignature()),
                args: value.getArgs().map((arg) => this.serializeValue(arg)),
            };
        } else if (value instanceof AbstractInvokeExpr) {
            throw new Error('Unhandled CallExpr: ' + util.inspect(value, { showHidden: true, depth: null }));
        } else if (value instanceof ArkThisRef) {
            return {
                _: 'ThisRef',
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkParameterRef) {
            return {
                _: 'ParameterRef',
                index: value.getIndex(),
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkArrayRef) {
            return {
                _: 'ArrayRef',
                array: this.serializeValue(value.getBase()),
                index: this.serializeValue(value.getIndex()),
                type: this.serializeType(value.getType()),
            };
        } else if (value instanceof ArkInstanceFieldRef) {
            return {
                _: 'InstanceFieldRef',
                instance: this.serializeValue(value.getBase()),
                field: this.serializeFieldSignature(value.getFieldSignature()),
            };
        } else if (value instanceof ArkStaticFieldRef) {
            return {
                _: 'StaticFieldRef',
                field: this.serializeFieldSignature(value.getFieldSignature()),
            };
        } else if (value instanceof AbstractFieldRef) {
            throw new Error('Unhandled FieldRef: ' + util.inspect(value, { showHidden: true, depth: null }));
        } else if (value instanceof AbstractRef) {
            throw new Error('Unhandled Ref: ' + util.inspect(value, { showHidden: true, depth: null }));
        } else if (value instanceof AbstractExpr) {
            throw new Error('Unhandled Expr: ' + util.inspect(value, { showHidden: true, depth: null }));
        } else {
            throw new Error('Unhandled Value: ' + util.inspect(value, { showHidden: true, depth: null }));
        }
    }

    private serializeStmt(stmt: Stmt): object {
        if (stmt instanceof ArkAssignStmt) {
            return {
                _: 'AssignStmt',
                left: this.serializeValue(stmt.getLeftOp()),
                right: this.serializeValue(stmt.getRightOp()),
            };
        } else if (stmt instanceof ArkInvokeStmt) {
            return {
                _: 'CallStmt',
                expr: this.serializeValue(stmt.getInvokeExpr()),
            };
        } else if (stmt instanceof ArkIfStmt) {
            return {
                _: 'IfStmt',
                condition: this.serializeValue(stmt.getConditionExpr()),
            };
        } else if (stmt instanceof ArkReturnVoidStmt) {
            return {
                _: 'ReturnVoidStmt',
            };
        } else if (stmt instanceof ArkReturnStmt) {
            return {
                _: 'ReturnStmt',
                arg: this.serializeValue(stmt.getOp()),
            };
        } else if (stmt instanceof ArkThrowStmt) {
            return {
                _: 'ThrowStmt',
                arg: this.serializeValue(stmt.getOp()),
            };
        } else {
            throw new Error('Unhandled Stmt: ' + util.inspect(stmt, { showHidden: true, depth: null }));
        }
    }
}
