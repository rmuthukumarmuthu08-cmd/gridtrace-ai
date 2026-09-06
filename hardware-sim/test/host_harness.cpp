// Host-side driver: compiles the REAL src/main.cpp against the stubs above and
// runs it for a few simulated hours, cycling through all four modes.
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <vector>
#include <string>

uint32_t g_fakeMillis = 0;
int      g_fakeAdc    = 2048;
int      g_pinState[64];
std::deque<int> g_serialIn;
std::deque<std::string> g_serialOut;
FakeSerial Serial;
TwoWire Wire;
std::vector<std::string> g_oledLines;

void setup();
void loop();

int main(int argc, char **argv) {
  for (int i = 0; i < 64; i++) g_pinState[i] = HIGH;  // buttons idle (pull-up)
  int ticksPerMode = (argc > 1) ? atoi(argv[1]) : 180;

  setup();

  const char cmds[4] = { '1', '2', '3', '4' };
  for (int m = 0; m < 4; m++) {
    g_serialIn.push_back(cmds[m]);
    // 20 loop iterations of 50 ms == 1 s == 1 telemetry tick
    for (int i = 0; i < ticksPerMode * 20; i++) {
      loop();
      g_fakeMillis += 50;
    }
  }
  return 0;
}
