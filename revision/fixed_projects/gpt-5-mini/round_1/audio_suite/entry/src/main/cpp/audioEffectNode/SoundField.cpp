/*
 * Copyright (c) 2025 Huawei Device Co., Ltd. 2025-2025. ALL rights reserved.
 */

#include "SoundField.h"

OH_SoundFieldType getSoundFieldTypeByNum(int mode)
{
    OH_SoundFieldType type;
    switch (mode) {
        case OH_SoundFieldType::SOUND_FIELD_FRONT_FACING:
            type = OH_SoundFieldType::SOUND_FIELD_FRONT_FACING;
            break;
        case OH_SoundFieldType::SOUND_FIELD_GRAND:
            type = OH_SoundFieldType::SOUND_FIELD_GRAND;
            break;
        case OH_SoundFieldType::SOUND_FIELD_NEAR:
            type = OH_SoundFieldType::SOUND_FIELD_NEAR;
            break;
        case OH_SoundFieldType::SOUND_FIELD_WIDE:
            type = OH_SoundFieldType::SOUND_FIELD_WIDE;
            break;
        default:
            type = OH_SoundFieldType::SOUND_FIELD_FRONT_FACING;
            break;
    }
    return type;
}