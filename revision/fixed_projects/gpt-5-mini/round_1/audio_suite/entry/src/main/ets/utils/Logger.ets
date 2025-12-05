/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

import hilog from '@ohos.hilog';

/**
 * Logger
 */
export class Logger {
    private static readonly PARENT_CONTROL_DOMAIN: number = 0x5555;
    private static readonly hpcString: string = 'HwParentControl';

    /**
     * debug log
     *
     * @param tag tag
     * @param msg message
     */
    public static debug(tag: string, msg: string, ...args: (Object | undefined | null)[]): void {
        const msgArg = Logger.formatLogArg(msg, ...args);
        hilog.debug(Logger.PARENT_CONTROL_DOMAIN, Logger.hpcString, `${tag} - ${msgArg}`);
    }

    /**
     * info log
     *
     * @param tag tag
     * @param msg message
     */
    public static info(tag: string, msg: string, ...args: (Object | undefined | null)[]): void {
        const msgArg = Logger.formatLogArgSafely(msg, ...args);
        hilog.info(Logger.PARENT_CONTROL_DOMAIN, Logger.hpcString, `${tag} - ${msgArg}`);
    }

    /**
     * warn log
     *
     * @param tag tag
     * @param msg message
     */
    public static warn(tag: string, msg: string, ...args: (Object | undefined | null)[]): void {
        const msgArg = Logger.formatLogArgSafely(msg, ...args);
        hilog.warn(Logger.PARENT_CONTROL_DOMAIN, Logger.hpcString, `${tag} - ${msgArg}`);
    }

    /**
     * error log
     *
     * @param tag tag
     * @param msg message
     */
    public static error(tag: string, msg: string, ...args: (Object | undefined | null)[]): void {
        const msgArg = Logger.formatLogArgSafely(msg, ...args);
        hilog.error(Logger.PARENT_CONTROL_DOMAIN, Logger.hpcString, `${tag} - ${msgArg}`);
    }

    private static formatLogArg(msg: string, ...args: (Object | undefined | null)[]): string {
        return msg.replace(/{\d+}/g, (match) => {
            const index = match.match(/\d+/)?.[0];
            if (args.length > 0 && index !== undefined) {
                return Logger.getPrintArg(args[index]);
            } else {
                return '';
            }
        })
    }

    private static formatLogArgSafely(msg: string, ...args: (Object | undefined | null)[]): string {
        return msg.replace(/{\d+}/g, (match) => {   
            const index = match.match(/\d+/)?.[0];
            if (args.length > 0 && index !== undefined) {
                return Logger.getSafetyPrintArg(args[index]);
            } else {
                return '';
            }
        })
    }

    /**
     * debug级别格式化日志： 字符串、数字、布尔值用原始打印，Error类型只打印名称与描述、其他的用JSON.stringify转换为字符串后打印
     * @param logs
     */
    private static getPrintArg(log: Object | undefined): string {
        try {
            if (typeof log !== 'object' && log!== null && log !== undefined) {
                return log;
            }
            if (log instanceof Error) {
                return `errorName: ${log.name}. errorMessage: ${log.message}`;
            }
            return JSON.stringify(log);
        } catch {
            return '{print.err}';
        }
    }

    /**
     * info、warn、error级别格式化日志：Error类型只打印名称与描述、禁止打印对象、禁止打印json对象
     * @param logs
     */
    private static getSafetyPrintArg(log: Object | undefined | null): string {
        try {
            if (log === null || log === undefined) {
                return 'undefined';
            }
            // 规则1： Error类型只打印名称与描述
            if (log instanceof Error) {
                return `errorName: ${log.name}. errorMessage: ${log.message}`;
            }
            // 规则2： 禁止打印对象
            if (typeof log === 'object') {
                return '{print.obj}';
            }
            // 规则3： 禁止打印json对象
            if (typeof log === 'string') {
                return Logger.replaceBracesWithLength(log);
            }
            return log;
        } catch {
            return '{print.err}';
        }
    }

    /**
     * 替换{}内容为长度， 防止打印json对象
     * @param input 输入字符串
     * @returns 替换后的字符串
     */
    private static replaceBracesWithLength(input: string): string {
        return input.replace(/\{.*\}/, (match) => {
            const length = match.length;
            return '{print.json.len: ' + length.toString() + '}';
        });
    }
}