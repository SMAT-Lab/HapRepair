/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include <cstring>
#include "CompareFile.h"
#include "hilog/log.h"

static const int GLOBAL_RESMGR = 0xFF00;
static const char *TAG = "[AudioEditTestApp_CompareFile_cpp]";

// 比较文件长度
bool ValidateFileLength(std::ifstream& file1, std::ifstream& file2)
{
    file1.seekg(0, std::ios::end);
    file2.seekg(0, std::ios::end);

    if (file1.tellg() != file2.tellg()) {
        OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---file length is not equal");
        return false;
    }

    file1.seekg(0, std::ios::beg);
    file2.seekg(0, std::ios::beg);

    return true;
}

// 比较文件内容
bool CompareFileContent(std::ifstream& file1, std::ifstream& file2)
{
    const size_t bufferSize = 4096; // 4KB 缓冲区
    char buffer1[bufferSize];
    char buffer2[bufferSize];

    while (file1.good() && file2.good()) {
        file1.read(buffer1, bufferSize);
        file2.read(buffer2, bufferSize);

        if (file1.gcount() != file2.gcount() ||
            std::memcmp(buffer1, buffer2, static_cast<size_t>(file1.gcount())) != 0) {
                OH_LOG_Print(LOG_APP, LOG_INFO, GLOBAL_RESMGR, TAG, "audioEditTest---files binary is not equal");
                return false;
        }
    }

    if (file1.bad() || file2.bad()) {
        OH_LOG_Print(LOG_APP, LOG_ERROR, GLOBAL_RESMGR, TAG, "audioEditTest---file read error");
        return false;
    }

    return true;
}