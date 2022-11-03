/*
 * Copyright 2022 SenseTime Group Limited
 *
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

#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/settings.h"

#define DEFAULT_RESERVE_COUNT 4

#define CAL_SETTING_SZ(count) (sizeof(struct _setting_t) + sizeof(setting_entry_t) * (count))

static void setting_expand(setting_t *setting)
{
    if (NULL == setting || NULL == *setting) {
        return;
    }
    size_t count = (*setting)->reserved ? (*setting)->reserved * 2 : 1;
    setting_t res = (setting_t)realloc(*setting, CAL_SETTING_SZ(count));
    if (NULL != res) {
        res->reserved = count;
        *setting = res;
    }
}

setting_t setting_new() { return setting_new_with_reserved(DEFAULT_RESERVE_COUNT); }

setting_t setting_new_with_reserved(size_t count)
{
    setting_t res = (setting_t)malloc(CAL_SETTING_SZ(count));
    memset(res, 0, sizeof(struct _setting_t));
    return res;
}

void setting_shrink(setting_t *setting)
{
    if (NULL == setting || NULL == *setting) {
        return;
    }
    if ((*setting)->len == (*setting)->reserved)
        return;
    setting_t res = (setting_t)realloc(*setting, CAL_SETTING_SZ((*setting)->len));
    if (NULL != res) {
        res->reserved = res->len;
        *setting = res;
    }
}

setting_entry_t *setting_search(setting_t setting, uint32_t item)
{
    if (NULL == setting)
        return NULL;
    size_t i;
    for (i = 0; i < setting->len; ++i) {
        if (setting->entries[i].item == item) {
            return &setting->entries[i];
        }
    }
    return NULL;
}

void setting_delete(setting_t setting)
{
    if (NULL == setting)
        return;
    size_t i;
    for (i = 0; i < setting->len; ++i) {
        switch (setting->entries[i].tp) {
        case ENUM_SETTING_VALUE_SINGLE:
            if (setting->entries[i].dtype == dtSTR)
                free(setting->entries[i].v.single.value.str);
            break;
        case ENUM_SETTING_VALUE_REPEATED:
            if (setting->entries[i].dtype == dtSTR) {
                size_t j;
                for (j = 0; j < setting->entries[i].v.repeated.len; ++j) {
                    free(((char **)setting->entries[i].v.repeated.values)[j]);
                }
            }
            free(setting->entries[i].v.repeated.values);
            break;
        }
    }
    free(setting);
}

void setting_set_single(setting_t *setting, uint32_t item, uint32_t dtype, uvalue_t v)
{
    setting_entry_t *entry = setting_search(*setting, item);
    if (NULL != entry) {
        if (ENUM_SETTING_VALUE_REPEATED == entry->tp) {
            if (entry->dtype == dtSTR) {
                size_t j;
                for (j = 0; j < entry->v.repeated.len; ++j) {
                    free(((char **)entry->v.repeated.values)[j]);
                }
            }
            free(entry->v.repeated.values);
        } else if (entry->dtype == dtSTR) {
            free(entry->v.single.value.str);
        }
    } else { /* NULL == entry */
        if ((*setting)->len >= (*setting)->reserved) {
            setting_expand(setting);
        }
        entry = &(*setting)->entries[(*setting)->len++];
    }
    entry->item = item;
    entry->dtype = dtype;
    entry->tp = ENUM_SETTING_VALUE_SINGLE;
    if (dtSTR == dtype && NULL != v.str) {
        char *str = (char *)malloc(strlen(v.str) + 1);
        strcpy(str, v.str);
        entry->v.single.value.str = str;
    } else {
        entry->v.single.value = v;
    }
}

void *setting_alloc_repeated(setting_t *setting, uint32_t item, uint32_t dtype, size_t count)
{
    setting_entry_t *entry = setting_search(*setting, item);
    if (NULL != entry) {
        if (ENUM_SETTING_VALUE_REPEATED == entry->tp) {
            if (entry->dtype == dtSTR) {
                size_t j;
                for (j = 0; j < entry->v.repeated.len; ++j) {
                    free(((char **)entry->v.repeated.values)[j]);
                }
            }
            free(entry->v.repeated.values);
        } else if (entry->dtype == dtSTR) {
            free(entry->v.single.value.str);
        }
    } else { /* NULL == entry */
        if ((*setting)->len >= (*setting)->reserved) {
            setting_expand(setting);
        }
        entry = &(*setting)->entries[(*setting)->len++];
    }
    entry->item = item;
    entry->dtype = dtype;
    entry->tp = ENUM_SETTING_VALUE_REPEATED;
    if (0 < count) {
        entry->v.repeated.values = (void *)malloc(datatype_sizeof(dtype) * count);
        CHECK_NE(NULL, entry->v.repeated.values);
        memset(entry->v.repeated.values, 0, datatype_sizeof(dtype) * count);
    } else
        entry->v.repeated.values = NULL;
    entry->v.repeated.len = count;
    return entry->v.repeated.values;
}
