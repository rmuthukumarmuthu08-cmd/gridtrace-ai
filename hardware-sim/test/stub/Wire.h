#pragma once
#include <cstdint>
struct TwoWire {
  void begin(int, int) {}
  void setClock(uint32_t) {}
  void setTimeOut(uint16_t) {}
  void beginTransmission(int) {}
  int  endTransmission() { return 0; }   // host stub: pretend the OLED ACKs
};
extern TwoWire Wire;
