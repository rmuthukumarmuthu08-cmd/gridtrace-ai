#pragma once
#include <Wire.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define SSD1306_WHITE 1
#define SSD1306_SWITCHCAPVCC 2

extern std::vector<std::string> g_oledLines;

struct Adafruit_SSD1306 {
  Adafruit_SSD1306(int, int, TwoWire *, int) {}
  bool begin(int, int, bool = true, bool = true) { return true; }
  void clearDisplay() { g_oledLines.clear(); }
  void setTextSize(int) {}
  void setTextColor(int) {}
  void setCursor(int, int) {}
  void println(const char *s) { g_oledLines.push_back(std::string(s)); }
  void print(const char *s)   { g_oledLines.push_back(std::string(s)); }
  void display() {}
};
