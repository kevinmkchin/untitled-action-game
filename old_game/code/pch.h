#pragma once

#include <cstdint>
#include <cassert>
#include <fstream>
#include <string>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <stack>
#include <memory>
#include <utility>
#include <thread>
#include <mutex>
#include <random>

#define NOMINMAX
#include <windows.h>
#include <dwmapi.h>
#include <direct.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_mixer/SDL_mixer.h>

typedef uint8_t       u8;
typedef uint16_t      u16;
typedef uint32_t      u32;
typedef uint64_t      u64;
typedef int8_t        i8;
typedef int16_t       i16;
typedef int32_t       i32;
typedef int64_t       i64;
typedef uint_fast8_t  u8f;
typedef uint_fast16_t u16f;
typedef uint_fast32_t u32f;
typedef int_fast8_t   i8f;
typedef int_fast16_t  i16f;
typedef int_fast32_t  i32f;
typedef i16           bool16;
typedef i32           bool32;

#if (defined _MSC_VER)
#define pch_ASSERT(predicate) if(!(predicate)) { __debugbreak(); }
#else
#define pch_ASSERT(predicate) if(!(predicate)) { __builtin_trap(); }
#endif

