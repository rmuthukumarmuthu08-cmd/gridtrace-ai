// Minimal host-side Arduino shim -- ONLY used to compile & verify src/main.cpp
// natively on a PC. Not part of the ESP32 firmware.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <string>
#include <deque>

#define INPUT_PULLUP 2
#define OUTPUT 1
#define INPUT 0
#define HIGH 1
#define LOW 0

extern uint32_t g_fakeMillis;
extern int      g_fakeAdc;
extern int      g_pinState[64];
extern std::deque<int> g_serialIn;
extern std::deque<std::string> g_serialOut;

inline uint32_t millis() { return g_fakeMillis; }
inline void delay(uint32_t ms) { g_fakeMillis += ms; }
inline void pinMode(int, int) {}
inline int  digitalRead(int p) { return g_pinState[p & 63]; }
inline void digitalWrite(int p, int v) { g_pinState[p & 63] = v; }
inline int  analogRead(int) { return g_fakeAdc; }
inline void analogReadResolution(int) {}

struct FakeSerial {
  void begin(long) {}
  int  available() { return (int)g_serialIn.size(); }
  int  read() { if (g_serialIn.empty()) return -1; int c = g_serialIn.front(); g_serialIn.pop_front(); return c; }
  void println(const char *s) { g_serialOut.push_back(std::string(s)); printf("%s\n", s); }
  void print(const char *s)   { printf("%s", s); }
};
extern FakeSerial Serial;
