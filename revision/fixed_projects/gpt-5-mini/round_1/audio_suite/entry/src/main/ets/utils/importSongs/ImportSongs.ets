/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

import { common } from '@kit.AbilityKit';
import { SongInfo } from './InterfaceInfo';
import { picker } from '@kit.CoreFileKit';
import { METADATA_SERVICE, MusicMetadata } from './MusicMetadataService';
import { Logger } from '../Logger';

const TAG: string = 'AudioEditTestApp_ImportSongs';

// 导入歌曲
export async function importSongs(playList: Map<string, SongInfo>): Promise<Map<string, SongInfo>> {
    let newMap: Map<string, SongInfo> = new Map();
    try {
        let context = getContext() as common.Context; // 请确保 getContext(this) 返回结果为 UIAbilityContext
        let documentPicker = new picker.DocumentViewPicker(context);
        let selectResult: string[] =
            await documentPicker.select({fileSuffixFilters: ['.wav'], maxSelectNumber: 500});
        Logger.info(TAG, `DocumentViewPicker.select successfully, documentSelectResult uri: ${JSON.stringify(selectResult)}`);
        let tempMap: Map<string, SongInfo> = new Map();
        playList.forEach((song: SongInfo, key: string) => {
            tempMap.set(key, song);
        });
        for (let i = 0; i < selectResult.length; i++) {
            let uri = selectResult[i];
            let index = uri.lastIndexOf('/') + 1;
            let fileName = decodeURIComponent(uri.slice(index));
            let fileType = fileName.substring(fileName.lastIndexOf('.') + 1);
            let info: SongInfo = {
                uri,
                duration: 0,
                songName: fileName,
                songType: fileType,
            };
            if (!tempMap.has(info.uri)) {
                await addToList(uri, info, tempMap);
            }
        }
        newMap = tempMap;
    } catch (error) {
        Logger.error(TAG, `DocumentViewPicker failed with err: ${JSON.stringify(error)}`);
    }
    return newMap;
}

// 导入歌曲（修改为单文件导入）
export async function importSingleSong(nodeList: Map<string, SongInfo>,
    nodeId: string): Promise<Map<string, SongInfo>> {
    let newMap: Map<string, SongInfo> = new Map();
    try {
        let context = getContext() as common.Context;
        let documentPicker = new picker.DocumentViewPicker(context);

        // 修改：设置 maxSelectNumber 为 1，限制只能选择一个文件
        let selectResult: string[] =
            await documentPicker.select({
                fileSuffixFilters: ['.pcm', '.wav'],
                maxSelectNumber: 1 // 限制只能选择1个文件
        });
        Logger.info(TAG, `DocumentViewPicker.select successfully, documentSelectResult uri: ${JSON.stringify(selectResult)}`);

        // 复制原始播放列表
        let tempMap: Map<string, SongInfo> = new Map(nodeList);

        // 只处理第一个文件（如果有）
        if (selectResult.length > 0) {
            let uri = selectResult[0]; //只取第一个文件
            let index = uri.lastIndexOf('/') + 1;
            let fileName = decodeURIComponent(uri.slice(index));
            let fileType = fileName.substring(fileName.lastIndexOf('.') + 1);

            let info: SongInfo = {
                uri,
                duration: 0,
                songName: fileName,
                songType: fileType,
                nodeId: nodeId
            };
            // 检查是否已存在
            await addToList(uri, info, tempMap, true);
        }
        Logger.warn(TAG, `File ${tempMap.size} ***************`);
        newMap = tempMap;
    } catch (error) {
        Logger.error(TAG, `DocumentViewPicker failed with err: JSON.stringify(error)`);
    }

    return newMap;
}

/**
 * 导入非重复歌曲
 * @param uri      导入文件的uri
 * @param info     已有歌曲信息
 * @param tempMap  所有歌曲集合的 map
 */
export async function addToList(uri: string, info: SongInfo, tempMap: Map<string, SongInfo>, isSingle?: boolean) {
    await METADATA_SERVICE.fetchMetadata(uri).then((data: MusicMetadata) => {
        if (data.metadata?.title) {
            info.songName = data.metadata.title;
        }
        if (data.metadata?.duration) {
            info.duration = Number(data.metadata.duration);
        }
        if (data.metadata?.artist) {
            info.singerName = data.metadata.artist;
        }
        info.albumName = data.metadata?.album;
        info.albumCover = data.pixelMap;
    }).finally(() => {
        if (isSingle && info?.nodeId) {
            tempMap.set(info.nodeId, info);
        } else {
            tempMap.set(info.uri, info);
        }
    });
}