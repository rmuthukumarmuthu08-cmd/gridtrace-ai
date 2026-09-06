/* ===========================================================================
 *  AI-Powered Real-Time Energy Loss Investigation & Localization System
 *  ---------------------------------------------------------------------
 *  STAGE 1 ONLY: HARDWARE / TELEMETRY SIMULATION (Wokwi + ESP32)
 *
 *  Author : Nirmal (Muthukumar R) - Team Squad Mavericks
 *  Target : ESP32 DevKit-C v4 (Wokwi simulation)
 *
 *  WHAT THIS IS
 *  ------------
 *  A synthetic smart-metering testbed. One ESP32 plays the role of a feeder
 *  data concentrator that reads 5 virtual consumer meters organised as:
 *
 *      Feeder F01
 *       |-- Zone Z01 : C001 (low res), C002 (med res), C003 (high res)
 *       |-- Zone Z02 : C004 (commercial), C005 (variable)
 *
 *  It emits NDJSON telemetry (one JSON object per line) that a later
 *  Python / MQTT / n8n stage can consume directly.
 *
 *  WHAT THIS IS NOT
 *  ----------------
 *  This is a *measurement* simulator. It only perturbs numbers in software.
 *  It contains no instructions, circuitry, or mechanism for tampering with,
 *  bypassing, or physically interfering with any real electricity meter or
 *  supply. Nothing here should ever be wired to mains. The "anomaly" modes
 *  exist purely to produce labelled test data for the future AI module, and
 *  an energy imbalance in the data does NOT by itself mean theft - it can
 *  equally be technical loss, a faulty meter, or a metering/data gap.
 * ===========================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------ pins */
#define PIN_I2C_SDA    21
#define PIN_I2C_SCL    22
#define PIN_BTN_MODE   25   /* to GND, INPUT_PULLUP */
#define PIN_BTN_VIEW   26   /* to GND, INPUT_PULLUP */
#define PIN_POT        34   /* ADC1_CH6, input-only pin  */
#define PIN_LED_OK     18
#define PIN_LED_ALERT  19

/* ------------------------------------------------------------------ oled */
#define SCREEN_W       128
#define SCREEN_H        64
#define OLED_RESET      -1
#define OLED_ADDR     0x3C
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, OLED_RESET);
static bool oledOk = false;

/* ------------------------------------------------------------ sim timing */
#define TICK_MS            1000UL   /* one telemetry frame per real second  */
#define OLED_MS             250UL
#define SIM_SECONDS_PER_TICK  60    /* 1 real sec = 1 simulated minute      */
                                    /* => a full 24 h load curve in 24 min  */
#define SIM_EPOCH_START  1788588000UL  /* 2026-09-05T06:00:00Z              */

/* ----------------------------------------------------------- grid params */
#define V_NOMINAL        233.0f   /* open-circuit / no-load bus volts       */
#define V_DROOP_PER_KW     0.42f  /* volts lost per kW of feeder load       */
#define V_MIN            200.0f
#define V_MAX            250.0f
#define TECH_LOSS_FRAC     0.032f /* baseline technical (I^2R) loss ~3.2%   */
#define DEV_THRESHOLD      0.35f  /* +/-35% vs EWMA baseline => flag        */

/* ------------------------------------------------------------ RNG (LCG)  */
/* Deterministic on purpose: reruns produce the same dataset, which makes   */
/* model training / debugging reproducible.                                 */
static uint32_t g_rng = 0xC0FFEEu;
static inline uint32_t rngNext() { g_rng = g_rng * 1664525u + 1013904223u; return g_rng; }
static inline float rnd01()  { return (float)(rngNext() >> 8) / 16777216.0f; }
static inline float rndSym() { return rnd01() * 2.0f - 1.0f; }   /* -1 .. +1 */

/* ------------------------------------------------------------------ mode */
enum SimMode {
  MODE_NORMAL = 0,
  MODE_ABNORMAL,
  MODE_PERSISTENT,
  MODE_IMBALANCE,
  MODE_COUNT
};
static const char *MODE_NAME[MODE_COUNT] = {
  "NORMAL", "ABNORMAL_CONSUMPTION", "PERSISTENT_ANOMALY", "NETWORK_IMBALANCE"
};
static const char *MODE_SHORT[MODE_COUNT] = { "NORMAL", "ABNORMAL", "PERSIST", "IMBALANCE" };
/* Ground-truth label written into every record for future supervised work. */
static const char *MODE_LABEL[MODE_COUNT] = {
  "normal", "abnormal_consumption", "persistent_deviation", "network_imbalance"
};
static uint8_t g_mode = MODE_NORMAL;

/* -------------------------------------------------------------- profiles */
enum Profile { PROF_RES_LOW = 0, PROF_RES_MED, PROF_RES_HIGH, PROF_COMMERCIAL, PROF_VARIABLE };

struct ConsumerCfg {
  const char *id;
  const char *zone;
  const char *feeder;
  const char *type;
  uint8_t     profile;
  float       baseKW;    /* load at shape == 1.0                */
  float       pfNom;     /* nominal power factor                */
  float       zoneDropV; /* extra volt drop for its zone         */
};

#define NUM_CONSUMERS 5
static const ConsumerCfg CFG[NUM_CONSUMERS] = {
  { "C001", "Z01", "F01", "residential_low",    PROF_RES_LOW,    0.45f, 0.97f, 1.4f },
  { "C002", "Z01", "F01", "residential_medium", PROF_RES_MED,    1.10f, 0.95f, 1.6f },
  { "C003", "Z01", "F01", "residential_high",   PROF_RES_HIGH,   2.20f, 0.93f, 1.9f },
  { "C004", "Z02", "F01", "commercial",         PROF_COMMERCIAL, 3.60f, 0.89f, 3.2f },
  { "C005", "Z02", "F01", "variable_load",      PROF_VARIABLE,   1.60f, 0.91f, 3.5f }
};

struct ConsumerState {
  float walk;            /* slow random walk around the profile   */
  float trueKW;          /* power actually delivered              */
  float measKW;          /* power the meter reports               */
  float voltage;
  float currentA;
  float pf;
  float trueEnergy;      /* kWh actually delivered                */
  float meterEnergy;     /* kWh registered by the meter           */
  float ewmaKW;          /* adaptive behavioural baseline         */
  bool  injected;        /* ground truth: anomaly injected here?  */
  const char *status;
};
static ConsumerState ST[NUM_CONSUMERS];

/* ---------------------------------------------------------- feeder state */
static float g_feederInputEnergy = 0.0f;   /* kWh measured at feeder head   */
static float g_feederInputKW     = 0.0f;
static float g_extraUnaccKW      = 0.0f;   /* simulated unaccounted block   */
static float g_gridScale         = 1.0f;   /* from potentiometer            */

/* -------------------------------------------------------------- runtime  */
static uint32_t g_tick       = 0;
static uint32_t g_simEpoch   = SIM_EPOCH_START;
static uint32_t g_lastTick   = 0;
static uint32_t g_lastOled   = 0;
static uint32_t g_lastPage   = 0;
static uint8_t  g_page       = 0;          /* 0..4 consumers, 5 = feeder    */
static bool     g_streamOn   = true;
static uint32_t g_lastBtnMode = 0;
static uint32_t g_lastBtnView = 0;

/* ===================================================================== */
/*  Load-shape helpers                                                   */
/* ===================================================================== */
static inline float gaussPk(float x, float mu, float sigma) {
  float z = (x - mu) / sigma;
  return expf(-0.5f * z * z);
}
static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

/* Returns a dimensionless multiplier ~0.2 .. ~1.6 for the hour of day. */
static float profileShape(uint8_t p, float h) {
  switch (p) {
    case PROF_RES_LOW:
      return 0.30f + 0.45f * gaussPk(h, 7.5f, 1.4f) + 0.85f * gaussPk(h, 20.0f, 2.0f);
    case PROF_RES_MED:
      return 0.35f + 0.60f * gaussPk(h, 8.0f, 1.6f) + 1.00f * gaussPk(h, 20.5f, 2.2f);
    case PROF_RES_HIGH:
      return 0.40f + 0.70f * gaussPk(h, 7.0f, 1.5f) + 1.10f * gaussPk(h, 21.0f, 2.4f)
                   + 0.30f * gaussPk(h, 14.0f, 2.0f);
    case PROF_COMMERCIAL: {
      float on  = 1.0f / (1.0f + expf(-(h -  9.0f) * 2.2f));
      float off = 1.0f / (1.0f + expf(-(h - 18.5f) * 2.2f));
      return 0.20f + 1.10f * (on - off);
    }
    case PROF_VARIABLE:
      return 0.45f + 0.55f * (0.5f + 0.5f * sinf(h * 0.9f)) + 0.35f * gaussPk(h, 12.0f, 1.2f);
  }
  return 1.0f;
}

/* ISO-8601 UTC timestamp from the simulated clock. */
static void simTimestamp(char *out, size_t n) {
  time_t t = (time_t)g_simEpoch;
  struct tm tmv;
  gmtime_r(&t, &tmv);
  /* the % keeps every field digit-bounded so the buffer can never truncate */
  snprintf(out, n, "%04d-%02d-%02dT%02d:%02d:%02dZ",
           (tmv.tm_year + 1900) % 10000, (tmv.tm_mon + 1) % 100, tmv.tm_mday % 100,
           tmv.tm_hour % 100, tmv.tm_min % 100, tmv.tm_sec % 100);
}
static float simHourFloat() {
  uint32_t secOfDay = g_simEpoch % 86400UL;
  return (float)secOfDay / 3600.0f;
}

/* ===================================================================== */
/*  Core physics + anomaly injection for one tick                        */
/* ===================================================================== */
static void simulateTick() {
  const float h      = simHourFloat();
  const float dtHour = (float)SIM_SECONDS_PER_TICK / 3600.0f;

  /* Potentiometer = overall grid demand scaler (0.60 .. 1.40). */
  int raw = analogRead(PIN_POT);
  float potNorm = (float)raw / 4095.0f;
  g_gridScale = 0.60f + 0.80f * clampf(potNorm, 0.0f, 1.0f);

  /* ---- pass 1: true (delivered) power per consumer ------------------- */
  float trueKW[NUM_CONSUMERS];
  float meterFactor[NUM_CONSUMERS];
  bool  injected[NUM_CONSUMERS];
  float totalTrueKW = 0.0f;

  for (int i = 0; i < NUM_CONSUMERS; i++) {
    ST[i].walk = clampf(ST[i].walk * 0.92f + rndSym() * 0.030f, -0.15f, 0.15f);
    float kw = CFG[i].baseKW * profileShape(CFG[i].profile, h)
                             * g_gridScale * (1.0f + ST[i].walk);
    trueKW[i]      = kw;
    meterFactor[i] = 1.0f;      /* meter registers exactly what flows      */
    injected[i]    = false;
  }

  /* ---- anomaly injection (software only, measurement-level) ---------- */
  g_extraUnaccKW = 0.0f;

  switch (g_mode) {
    case MODE_NORMAL:
      break;

    case MODE_ABNORMAL: {
      /* C003: recurring sudden surge (~2.4x) for part of every cycle.    */
      uint32_t ph = g_tick % 40;
      if (ph < 15) { trueKW[2] *= 2.40f; injected[2] = true; }
      /* C005: unexpected sharp reduction in registered consumption.      */
      if ((g_tick % 37) < 5) { meterFactor[4] = 0.25f; injected[4] = true; }
      break;
    }

    case MODE_PERSISTENT: {
      /* C002: meter persistently under-registers (~45% of true flow).    */
      meterFactor[1] = 0.45f; injected[1] = true;
      /* C004: commercial site drawing load at an implausible hour.       */
      if (h < 5.0f || h > 23.0f) { trueKW[3] += 2.20f; injected[3] = true; }
      break;
    }

    case MODE_IMBALANCE:
      /* Consumer meters all look fine; the gap appears only at the       */
      /* feeder head. Deliberately ambiguous - could be loss, an          */
      /* unmetered connection, or a data gap. The AI stage must decide.   */
      break;
  }

  for (int i = 0; i < NUM_CONSUMERS; i++) totalTrueKW += trueKW[i];

  if (g_mode == MODE_IMBALANCE) {
    float frac = 0.13f + 0.04f * sinf((float)g_tick * 0.05f);   /* 9%..17% */
    g_extraUnaccKW = frac * totalTrueKW;
  }

  /* ---- pass 2: bus voltage falls as feeder loading rises ------------- */
  float busV = V_NOMINAL - V_DROOP_PER_KW * totalTrueKW + rndSym() * 0.35f;
  busV = clampf(busV, V_MIN, V_MAX);

  /* ---- pass 3: per-consumer electrical quantities -------------------- */
  for (int i = 0; i < NUM_CONSUMERS; i++) {
    float v  = clampf(busV - CFG[i].zoneDropV + rndSym() * 0.25f, V_MIN, V_MAX);
    float pf = clampf(CFG[i].pfNom + rndSym() * 0.015f, 0.75f, 1.00f);

    float measKW = trueKW[i] * meterFactor[i];
    /* I = P / (V * pf)   ->   keeps P = V * I * pf exactly consistent.   */
    float amps   = (measKW * 1000.0f) / (v * pf);

    ST[i].voltage  = v;
    ST[i].pf       = pf;
    ST[i].trueKW   = trueKW[i];
    ST[i].measKW   = measKW;
    ST[i].currentA = amps;
    ST[i].injected = injected[i];

    ST[i].trueEnergy  += trueKW[i] * dtHour;
    ST[i].meterEnergy += measKW    * dtHour;

    if (g_tick == 0) ST[i].ewmaKW = measKW;
    float dev = (ST[i].ewmaKW > 0.05f) ? (measKW - ST[i].ewmaKW) / ST[i].ewmaKW : 0.0f;
    ST[i].ewmaKW = 0.97f * ST[i].ewmaKW + 0.03f * measKW;

    if      (dev >  DEV_THRESHOLD) ST[i].status = "SURGE";
    else if (dev < -DEV_THRESHOLD) ST[i].status = "DIP";
    else                           ST[i].status = "OK";
  }

  /* ---- pass 4: feeder-head energy balance ---------------------------- */
  g_feederInputKW      = totalTrueKW * (1.0f + TECH_LOSS_FRAC) + g_extraUnaccKW;
  g_feederInputEnergy += g_feederInputKW * dtHour;
}

/* ===================================================================== */
/*  NDJSON telemetry                                                     */
/* ===================================================================== */
static char jbuf[768];

static void emitTelemetry() {
  if (!g_streamOn) return;

  char ts[32];
  simTimestamp(ts, sizeof(ts));
  const char *mode  = MODE_NAME[g_mode];
  const char *label = MODE_LABEL[g_mode];

  /* ---- consumer records -------------------------------------------- */
  float sumMeterEnergy = 0.0f;
  float zoneKW[2]      = { 0.0f, 0.0f };
  float zoneEnergy[2]  = { 0.0f, 0.0f };

  for (int i = 0; i < NUM_CONSUMERS; i++) {
    sumMeterEnergy += ST[i].meterEnergy;
    int z = (CFG[i].zone[2] == '1') ? 0 : 1;
    zoneKW[z]     += ST[i].measKW;
    zoneEnergy[z] += ST[i].meterEnergy;

    snprintf(jbuf, sizeof(jbuf),
      "{\"record_type\":\"consumer\",\"schema\":\"els.v1\",\"timestamp\":\"%s\","
      "\"uptime_ms\":%lu,\"tick\":%lu,"
      "\"consumer_id\":\"%s\",\"zone_id\":\"%s\",\"feeder_id\":\"%s\",\"load_type\":\"%s\","
      "\"voltage\":%.2f,\"current\":%.3f,\"power\":%.1f,\"power_factor\":%.3f,"
      "\"energy\":%.4f,\"status\":\"%s\",\"meter_status\":\"ONLINE\","
      "\"mode\":\"%s\",\"label\":\"%s\",\"anomaly_injected\":%s}",
      ts, (unsigned long)millis(), (unsigned long)g_tick,
      CFG[i].id, CFG[i].zone, CFG[i].feeder, CFG[i].type,
      ST[i].voltage, ST[i].currentA, ST[i].measKW * 1000.0f, ST[i].pf,
      ST[i].meterEnergy, ST[i].status,
      mode, label, ST[i].injected ? "true" : "false");
    Serial.println(jbuf);
  }

  /* ---- zone roll-ups ------------------------------------------------ */
  const char *zoneIds[2] = { "Z01", "Z02" };
  for (int z = 0; z < 2; z++) {
    snprintf(jbuf, sizeof(jbuf),
      "{\"record_type\":\"zone\",\"schema\":\"els.v1\",\"timestamp\":\"%s\","
      "\"uptime_ms\":%lu,\"tick\":%lu,"
      "\"zone_id\":\"%s\",\"feeder_id\":\"F01\","
      "\"metered_power\":%.1f,\"metered_energy\":%.4f,"
      "\"mode\":\"%s\",\"label\":\"%s\"}",
      ts, (unsigned long)millis(), (unsigned long)g_tick,
      zoneIds[z], zoneKW[z] * 1000.0f, zoneEnergy[z], mode, label);
    Serial.println(jbuf);
  }

  /* ---- feeder energy balance --------------------------------------- */
  float unacc    = g_feederInputEnergy - sumMeterEnergy;
  float unaccPct = (g_feederInputEnergy > 0.0001f)
                     ? (100.0f * unacc / g_feederInputEnergy) : 0.0f;

  /* INSTANTANEOUS gap. The cumulative figure above is diluted by every hour of
   * history, so a meter that has run for a week barely reacts to a new gap.
   * This one is history-free and is the signal an anomaly detector wants.   */
  float sumMeterKW  = zoneKW[0] + zoneKW[1];
  float unaccKW     = g_feederInputKW - sumMeterKW;
  float unaccKWPct  = (g_feederInputKW > 0.001f)
                        ? (100.0f * unaccKW / g_feederInputKW) : 0.0f;

  snprintf(jbuf, sizeof(jbuf),
    "{\"record_type\":\"feeder\",\"schema\":\"els.v1\",\"timestamp\":\"%s\","
    "\"uptime_ms\":%lu,\"tick\":%lu,"
    "\"feeder_id\":\"F01\",\"consumers\":%d,"
    "\"input_power\":%.1f,\"input_energy\":%.4f,"
    "\"sum_meter_energy\":%.4f,\"unaccounted_energy\":%.4f,\"unaccounted_pct\":%.2f,"
    "\"unaccounted_power\":%.1f,\"unaccounted_power_pct\":%.2f,"
    "\"technical_loss_pct_expected\":%.2f,\"grid_load_scale\":%.2f,"
    "\"mode\":\"%s\",\"label\":\"%s\","
    "\"note\":\"SIMULATED NETWORK IMBALANCE - unaccounted energy is not proof of theft\"}",
    ts, (unsigned long)millis(), (unsigned long)g_tick,
    NUM_CONSUMERS, g_feederInputKW * 1000.0f, g_feederInputEnergy,
    sumMeterEnergy, unacc, unaccPct,
    unaccKW * 1000.0f, unaccKWPct,
    100.0f * TECH_LOSS_FRAC / (1.0f + TECH_LOSS_FRAC), g_gridScale,
    mode, label);
  Serial.println(jbuf);
}

/* ===================================================================== */
/*  OLED                                                                 */
/* ===================================================================== */
static void drawOled() {
  if (!oledOk) return;
  char l[26];

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  if (g_page < NUM_CONSUMERS) {
    int i = g_page;
    snprintf(l, sizeof(l), "%s/%s  %s", CFG[i].feeder, CFG[i].zone, CFG[i].id);
    display.println(l);
    snprintf(l, sizeof(l), "MODE %s", MODE_SHORT[g_mode]);          display.println(l);
    snprintf(l, sizeof(l), "V  %6.1f V",  ST[i].voltage);           display.println(l);
    snprintf(l, sizeof(l), "I  %6.2f A",  ST[i].currentA);          display.println(l);
    snprintf(l, sizeof(l), "P  %6.0f W",  ST[i].measKW * 1000.0f);  display.println(l);
    snprintf(l, sizeof(l), "E  %6.2f kWh", ST[i].meterEnergy);      display.println(l);
    snprintf(l, sizeof(l), "PF %6.2f",    ST[i].pf);                display.println(l);
    snprintf(l, sizeof(l), "ST %s", ST[i].status);                  display.print(l);
  } else {
    float sum = 0.0f;
    for (int i = 0; i < NUM_CONSUMERS; i++) sum += ST[i].meterEnergy;
    float unacc    = g_feederInputEnergy - sum;
    float unaccPct = (g_feederInputEnergy > 0.0001f)
                       ? (100.0f * unacc / g_feederInputEnergy) : 0.0f;

    display.println("FEEDER F01 BALANCE");
    snprintf(l, sizeof(l), "MODE %s", MODE_SHORT[g_mode]);         display.println(l);
    snprintf(l, sizeof(l), "IN  %7.3f kWh", g_feederInputEnergy);  display.println(l);
    snprintf(l, sizeof(l), "MTR %7.3f kWh", sum);                  display.println(l);
    snprintf(l, sizeof(l), "GAP %7.3f kWh", unacc);                display.println(l);
    snprintf(l, sizeof(l), "GAP %7.2f %%",  unaccPct);             display.println(l);
    if (g_mode == MODE_IMBALANCE) {
      display.println("SIMULATED NETWORK");
      display.print(  "IMBALANCE");
    } else {
      display.println("baseline tech loss");
      display.print(  "~3.1 % expected");
    }
  }
  display.display();
}

/* ===================================================================== */
/*  Inputs                                                               */
/* ===================================================================== */
static void setMode(uint8_t m) {
  g_mode = m % MODE_COUNT;
  snprintf(jbuf, sizeof(jbuf),
    "{\"record_type\":\"event\",\"schema\":\"els.v1\",\"uptime_ms\":%lu,"
    "\"event\":\"MODE_CHANGE\",\"mode\":\"%s\",\"mode_index\":%u}",
    (unsigned long)millis(), MODE_NAME[g_mode], (unsigned)g_mode);
  Serial.println(jbuf);
}

static void resetEnergy() {
  for (int i = 0; i < NUM_CONSUMERS; i++) {
    ST[i].trueEnergy = 0.0f;
    ST[i].meterEnergy = 0.0f;
  }
  g_feederInputEnergy = 0.0f;
  Serial.println("{\"record_type\":\"event\",\"schema\":\"els.v1\",\"event\":\"ENERGY_RESET\"}");
}

static void handleButtons() {
  uint32_t now = millis();
  if (digitalRead(PIN_BTN_MODE) == LOW && (now - g_lastBtnMode) > 250) {
    g_lastBtnMode = now;
    setMode(g_mode + 1);
  }
  if (digitalRead(PIN_BTN_VIEW) == LOW && (now - g_lastBtnView) > 250) {
    g_lastBtnView = now;
    g_page = (g_page + 1) % (NUM_CONSUMERS + 1);
    g_lastPage = now;
    drawOled();
  }
}

static void handleSerial() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    switch (c) {
      case '1': setMode(MODE_NORMAL);     break;
      case '2': setMode(MODE_ABNORMAL);   break;
      case '3': setMode(MODE_PERSISTENT); break;
      case '4': setMode(MODE_IMBALANCE);  break;
      case 'm': case 'M': setMode(g_mode + 1); break;
      case 'n': case 'N': g_page = (g_page + 1) % (NUM_CONSUMERS + 1); break;
      case 'p': case 'P': g_streamOn = !g_streamOn; break;
      case 'r': case 'R': resetEnergy();  break;
      default: break;
    }
  }
}

/* ===================================================================== */
/*  Arduino entry points                                                 */
/* ===================================================================== */
static bool g_oledTried = false;

/* Deferred display bring-up. Deliberately NOT in setup(): an unwired or
 * unresponsive I2C device must never be able to delay - let alone block -
 * the telemetry stream, which is the whole point of this firmware.        */
static void tryInitOled() {
  g_oledTried = true;
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setTimeOut(50);
  /* Scan both addresses the SSD1306 ships with: 0x3C (most 128x64 modules)
   * and 0x3D (Adafruit breakouts with SA0 pulled high). Assuming one of them
   * is the classic reason a perfectly good display stays blank.            */
  uint8_t addr = 0;
  const uint8_t candidates[2] = { 0x3C, 0x3D };
  for (int i = 0; i < 2 && addr == 0; i++) {
    Wire.beginTransmission(candidates[i]);
    if (Wire.endTransmission() == 0) addr = candidates[i];
  }
  /* reset=false, periphBegin=false -> do not let the library re-init I2C */
  oledOk = (addr != 0) && display.begin(SSD1306_SWITCHCAPVCC, addr, false, false);
  snprintf(jbuf, sizeof(jbuf),
    "{\"record_type\":\"event\",\"schema\":\"els.v1\",\"event\":\"OLED_INIT\","
    "\"i2c_address\":\"0x%02X\",\"present\":%s,\"ok\":%s}",
    addr, addr ? "true" : "false", oledOk ? "true" : "false");
  Serial.println(jbuf);
}

static const char *META_JSON =
  "{\"record_type\":\"meta\",\"schema\":\"els.v1\",\"device\":\"ESP32-WOKWI-SIM\","
  "\"feeder\":\"F01\",\"zones\":[\"Z01\",\"Z02\"],"
  "\"consumers\":[\"C001\",\"C002\",\"C003\",\"C004\",\"C005\"],"
  "\"tick_ms\":1000,\"sim_seconds_per_tick\":60,"
  "\"modes\":[\"NORMAL\",\"ABNORMAL_CONSUMPTION\",\"PERSISTENT_ANOMALY\",\"NETWORK_IMBALANCE\"],"
  "\"disclaimer\":\"Synthetic measurement data. No real electrical connection. "
  "Anomalies are software perturbations for AI testing only.\"}";

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PIN_BTN_MODE,  INPUT_PULLUP);
  pinMode(PIN_BTN_VIEW,  INPUT_PULLUP);
  pinMode(PIN_LED_OK,    OUTPUT);
  pinMode(PIN_LED_ALERT, OUTPUT);
  analogReadResolution(12);
  /* Boot heartbeat: if this LED is on, setup() started. Nothing before
   * this point can block, so it is a reliable 'the firmware booted' sign. */
  digitalWrite(PIN_LED_OK, HIGH);

  for (int i = 0; i < NUM_CONSUMERS; i++) {
    ST[i].walk = 0.0f;  ST[i].trueEnergy = 0.0f; ST[i].meterEnergy = 0.0f;
    ST[i].ewmaKW = 0.0f; ST[i].status = "OK";    ST[i].injected = false;
    ST[i].voltage = V_NOMINAL; ST[i].currentA = 0.0f; ST[i].pf = CFG[i].pfNom;
    ST[i].trueKW = 0.0f; ST[i].measKW = 0.0f;
  }

  Serial.println(META_JSON);

  g_lastTick = millis() - TICK_MS;
}

void loop() {
  uint32_t now = millis();

  handleButtons();
  handleSerial();

  if ((uint32_t)(now - g_lastTick) >= TICK_MS) {
    g_lastTick += TICK_MS;
    simulateTick();
    emitTelemetry();
    g_simEpoch += SIM_SECONDS_PER_TICK;
    g_tick++;

    bool alert = (g_mode != MODE_NORMAL);
    digitalWrite(PIN_LED_OK,    alert ? LOW : HIGH);
    digitalWrite(PIN_LED_ALERT, alert ? ((g_tick & 1) ? HIGH : LOW) : LOW);
  }

  if (!g_oledTried && g_tick >= 1) tryInitOled();

  if ((uint32_t)(now - g_lastPage) >= 3000UL) {   /* auto-advance OLED page */
    g_lastPage = now;
    g_page = (g_page + 1) % (NUM_CONSUMERS + 1);
  }

  if ((uint32_t)(now - g_lastOled) >= OLED_MS) {
    g_lastOled = now;
    drawOled();
  }
}
