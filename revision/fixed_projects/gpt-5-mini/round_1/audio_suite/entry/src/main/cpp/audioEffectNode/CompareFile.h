/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#ifndef TEST1_COMPAREFILE_H
#define TEST1_COMPAREFILE_H

#include <fstream>

class CompareFile {
};

bool ValidateFileLength(std::ifstream& file1, std::ifstream& file2);

bool CompareFileContent(std::ifstream& file1, std::ifstream& file2);

#endif //TEST1_COMPAREFILE_H