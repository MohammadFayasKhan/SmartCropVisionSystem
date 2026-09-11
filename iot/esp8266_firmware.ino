/*
 * ═══════════════════════════════════════════════════════════════════════
 *   Smart Plant Intelligence System: ESP8266 Firmware v2.0
 *   16x2 I2C LCD Edition (HW-61 / PCF8574 adapter)
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   HARDWARE:
 *     - NodeMCU ESP8266 (or Wemos D1 Mini)
 *     - DHT11 Temperature + Humidity Sensor (3-pin module)
 *     - Soil Moisture Sensor (AO analog output)
 *     - Rain Drop Detection Sensor (DO digital output)
 *     - 16x2 LCD with HW-61 I2C adapter (PCF8574 chip)
 *
 * ─────────────────────────────────────────────────────────────────────
 *   WIRING DIAGRAM
 * ─────────────────────────────────────────────────────────────────────
 *
 *   ESP8266 NodeMCU          16x2 LCD (HW-61 I2C back module)
 *   ───────────────          ────────────────────────────────
 *   D1 (GPIO5)  ─────────── SCL
 *   D2 (GPIO4)  ─────────── SDA
 *   VIN (5V)    ─────────── VCC   ← use VIN, not 3.3V (LCD needs 5V)
 *   GND         ─────────── GND
 *
 *   ESP8266 NodeMCU          DHT11 Module (3-pin: GND | DATA | VCC)
 *   ───────────────          ─────────────────────────────────────
 *   D4 (GPIO2)  ─────────── DATA  (middle pin on your module)
 *   3.3V        ─────────── VCC   (right pin on your module)
 *   GND         ─────────── GND   (left pin on your module)
 *   NOTE: Your module ALREADY has a pull-up resistor built in.
 *         Do NOT add an external resistor.
 *
 *   ESP8266 NodeMCU          Soil Moisture Sensor (AO|DO|GND|VCC)
 *   ───────────────          ─────────────────────────────────────
 *   A0          ─────────── AO   (analog output: most accurate)
 *   3.3V        ─────────── VCC
 *   GND         ─────────── GND
 *   [DO pin not connected: we use analog for better readings]
 *
 *   ESP8266 NodeMCU          Rain Sensor (AO|DO|GND|VCC)
 *   ───────────────          ────────────────────────────
 *   D5 (GPIO14) ─────────── DO   (digital output: dry/wet flag)
 *   3.3V        ─────────── VCC
 *   GND         ─────────── GND
 *   [AO pin not connected: only one ADC on ESP8266, used for soil]
 *
 * ─────────────────────────────────────────────────────────────────────
 *   REQUIRED LIBRARIES: install via Arduino IDE Library Manager
 * ─────────────────────────────────────────────────────────────────────
 *   1. ESP8266WiFi       : built-in with ESP8266 board package
 *   2. ESP8266HTTPClient : built-in with ESP8266 board package
 *   3. ArduinoJson       : v6.x by Benoit Blanchon
 *   4. DHT sensor library: by Adafruit
 *   5. LiquidCrystal I2C : by Frank de Brabander
 *      (Search "LiquidCrystal I2C" in Library Manager)
 *
 * ─────────────────────────────────────────────────────────────────────
 *   SETUP STEPS
 * ─────────────────────────────────────────────────────────────────────
 *   1. Fill in WIFI_SSID, WIFI_PASS below
 *   2. Find your Mac's local IP:
 *      Open Terminal → type:  ifconfig | grep "inet " | grep -v 127
 *      It will show something like: inet 192.168.1.105
 *   3. Set SERVER_IP to that address
 *   4. Make sure FastAPI is running:
 *      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
 *   5. If LCD shows garbled chars or nothing:
 *      Try changing LCD_I2C_ADDR from 0x27 to 0x3F
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

// ── Library Includes ───────────────────────────────────────────────────
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>      // for HTTPS / ngrok
#include <ArduinoJson.h>
#include <DHT11.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ══════════════════════════════════════════════════════════════════════
//   ⚙️  CONFIGURATION: EDIT THESE VALUES
// ══════════════════════════════════════════════════════════════════════

const char* WIFI_SSID   = "YOUR_WIFI_SSID";      // WiFi hotspot name
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";            // WiFi password

// ── Deployment Mode ─────────────────────────────────────────────────
//
//   MODE 1: LOCAL (default, fastest, no internet needed)
//     ESP8266 → same WiFi network → Mac → FastAPI on port 8000
//     Set USE_CLOUD false and USE_NGROK false.
//
//   MODE 2: HF SPACES / CLOUD (recommended for always-on deployment)
//     ESP8266 → Internet → Hugging Face Space → FastAPI on port 7860
//     Set USE_CLOUD true and fill in HF_SPACE_URL.
//     Your HF Space URL looks like: https://fayas-smart-plant.hf.space
//
//   MODE 3: NGROK (temporary tunnel from local Mac)
//     ESP8266 → Internet → ngrok → Mac → FastAPI
//     Set USE_NGROK true (and USE_CLOUD false).
//
// ─────────────────────────────────────────────────────────────────────
#define USE_CLOUD true    // ✅ HF Spaces: always-on cloud
#define USE_NGROK false   //    ngrok fallback (keep false)

// LOCAL mode: Mac IP on same WiFi/hotspot (only used when USE_CLOUD false)
const char* SERVER_IP   = "192.168.1.100";   // ← your Mac hotspot IP
const int   SERVER_PORT = 8000;

// HF SPACES URL: active because USE_CLOUD true
const char* HF_SPACE_URL = "https://your-username-smart-plant.hf.space";

// NGROK URL: only used when USE_NGROK true
const char* NGROK_URL = "https://your-tunnel-subdomain.ngrok-free.dev";

// ── LCD I2C Address ───────────────────────────────────────────────────
//   Most HW-61 modules use 0x27. If display is blank, try 0x3F.
#define LCD_I2C_ADDR    0x27
#define LCD_COLS        16
#define LCD_ROWS        2

// ── Pin Definitions ───────────────────────────────────────────────────
// GPIO numbers: more reliable than D-pin aliases across all ESP8266 boards
#define DHT_PIN         2      // GPIO2  = NodeMCU D4 : DHT11 DATA
#define RAIN_PIN        14     // GPIO14 = NodeMCU D5 : Rain sensor DO
#define SOIL_PIN        A0     // A0 (ADC)            : Soil moisture AO
#define I2C_SDA         4      // GPIO4  = NodeMCU D2 : LCD SDA
#define I2C_SCL         5      // GPIO5  = NodeMCU D1 : LCD SCL

// ── Timing ────────────────────────────────────────────────────────────
#define SEND_INTERVAL   15000  // POST to server every 15 seconds
#define SCREEN_INTERVAL  4000  // Rotate LCD screen every 4 seconds

// ── Soil Moisture Calibration ─────────────────────────────────────────
//   With your soil sensor: dry air ≈ 1023, wet/submerged ≈ 300
//   Adjust these if your sensor reads differently
#define SOIL_DRY_RAW    1023
#define SOIL_WET_RAW    300

// ─────────────────────────────────────────────────────────────────────

// ── Object Instances ───────────────────────────────────────────────────
DHT11              dht11(DHT_PIN);
LiquidCrystal_I2C  lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);
WiFiClient         plainClient;          // used in local HTTP mode
BearSSL::WiFiClientSecure secureClient; // used in ngrok HTTPS mode

// ── Global Sensor State ────────────────────────────────────────────────
int   g_temp        = 0;    // DHT11.h returns int (whole degrees)
int   g_hum         = 0;    // DHT11.h returns int (whole % RH)
int   g_soil        = 0;    // 0-100%
bool  g_rainOn      = false; // YL-83 digital output: true = rain detected
// NOTE: No rainfall in mm. The YL-83 sensor only outputs a binary wet/dry signal.

// ── Global Prediction State ────────────────────────────────────────────
String g_crop       = "";
float  g_conf       = 0.0;
String g_crop2      = "";
float  g_conf2      = 0.0;
String g_crop3      = "";
float  g_conf3      = 0.0;
int    g_alerts     = 0;
String g_alertName[4];
String g_alertSev[4];
int    g_alertCount = 0;
bool   g_serverOK   = false;

// ── Display State ──────────────────────────────────────────────────────
int           g_screen     = 0;
int           g_alertPage  = 0;    // which alert to show on screen 2
unsigned long g_lastSend   = 0;
unsigned long g_lastScreen = 0;

// ── Custom LCD characters ──────────────────────────────────────────────
byte degreeChar[8] = {
  0b00110, 0b01001, 0b01001, 0b00110,
  0b00000, 0b00000, 0b00000, 0b00000
};
byte tickChar[8] = {
  0b00000, 0b00001, 0b00011, 0b10110,
  0b11100, 0b01000, 0b00000, 0b00000
};
byte alertChar[8] = {
  0b00100, 0b01110, 0b01110, 0b01110,
  0b11111, 0b00000, 0b00100, 0b00000
};
byte dropChar[8] = {
  0b00100, 0b00100, 0b01110, 0b01110,
  0b11111, 0b11111, 0b01110, 0b00000
};

// ── Char indices ───────────────────────────────────────────────────────
#define CHAR_DEG    0
#define CHAR_TICK   1
#define CHAR_ALERT  2
#define CHAR_DROP   3

// ══════════════════════════════════════════════════════════════════════
//   SETUP
// ══════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println(F("\n\n=== Smart Plant Intelligence System v2.0 ==="));

  // ── DHT11 ─────────────────────────────────────────────────────────
  // DHT11.h does not require a begin() call: sensor is ready on first read

  // ── Rain sensor ───────────────────────────────────────────────────
  pinMode(RAIN_PIN, INPUT);

  // ── I2C LCD ───────────────────────────────────────────────────────
  Wire.begin(I2C_SDA, I2C_SCL);    // SDA = GPIO4, SCL = GPIO5
  lcd.init();
  lcd.backlight();
  lcd.createChar(CHAR_DEG,   degreeChar);
  lcd.createChar(CHAR_TICK,  tickChar);
  lcd.createChar(CHAR_ALERT, alertChar);
  lcd.createChar(CHAR_DROP,  dropChar);

  showBoot();

  // ── Connect WiFi ──────────────────────────────────────────────────
  connectWiFi();
}

// ══════════════════════════════════════════════════════════════════════
//   MAIN LOOP
// ══════════════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  // Re-connect WiFi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    g_serverOK = false;
    connectWiFi();
  }

  // Read sensors + POST to server every SEND_INTERVAL
  if (now - g_lastSend >= SEND_INTERVAL || g_lastSend == 0) {
    g_lastSend = now;
    readSensors();
    sendToServer();
  }

  // Rotate LCD screen every SCREEN_INTERVAL
  if (now - g_lastScreen >= SCREEN_INTERVAL) {
    g_lastScreen = now;
    advanceScreen();
    drawScreen();
  }
}

// ══════════════════════════════════════════════════════════════════════
//   READ ALL SENSORS
// ══════════════════════════════════════════════════════════════════════
void readSensors() {
  // ── DHT11 (DHT11.h library) ──────────────────────────────────────
  // Error 253 = "timed out": DHT11 needs at least 1-2s between reads.
  // We try once, wait 2s on failure, then try once more.
  // If both fail, we silently keep the previous valid reading so the
  // display and server always get a sensible value.
  int rawTemp = 0, rawHum = 0;
  int dhtResult = dht11.readTemperatureHumidity(rawTemp, rawHum);

  if (dhtResult != 0) {
    Serial.printf("DHT11 attempt 1 failed (%d), retrying in 2s...\n", dhtResult);
    delay(2000);   // DHT11 mandatory recovery time
    dhtResult = dht11.readTemperatureHumidity(rawTemp, rawHum);
  }

  if (dhtResult == 0) {
    g_temp = rawTemp;
    g_hum  = rawHum;
  } else {
    Serial.print(F("DHT11 both attempts failed: "));
    Serial.println(DHT11::getErrorString(dhtResult));
    // Keep previous valid readings: do not update g_temp / g_hum
  }

  // ── Soil Moisture (analog A0) ──────────────────────────────────────
  //   Raw ~1023 = bone dry, ~300 = saturated wet
  //   map() inverts so 0% = dry, 100% = wet
  int raw  = analogRead(SOIL_PIN);
  g_soil   = map(raw, SOIL_DRY_RAW, SOIL_WET_RAW, 0, 100);
  g_soil   = constrain(g_soil, 0, 100);

  // ── Rain Sensor (digital D5) ───────────────────────────────────────
  //   LOW = rain detected, HIGH = dry (standard YL-83 behaviour)
  //   The sensor outputs a pure binary signal: no mm estimation needed.
  g_rainOn = (digitalRead(RAIN_PIN) == LOW);

  Serial.printf(
    "T:%d H:%d Soil:%d Rain:%s\n",
    g_temp, g_hum, g_soil, g_rainOn ? "YES" : "NO"
  );
}

// ══════════════════════════════════════════════════════════════════════
//   SEND TO SERVER + PARSE RESPONSE
// ══════════════════════════════════════════════════════════════════════
void sendToServer() {
  lcdStatus("Sending...", "");

  // ── Build JSON payload (rain = binary 0 or 1) ─────────────────
  StaticJsonDocument<200> payload;
  payload["temperature"]   = g_temp;
  payload["humidity"]      = g_hum;
  payload["soil_moisture"] = g_soil;
  payload["rain"]          = g_rainOn ? 1 : 0;   // YL-83: 1=wet, 0=dry
  String body;
  serializeJson(payload, body);

  HTTPClient http;
  int code = -1;

#if USE_CLOUD
  // ── HTTPS to Hugging Face Space (always-on cloud) ─────────────────
  secureClient.setBufferSizes(512, 512);
  secureClient.setInsecure();   // skip cert chain: fine for IoT sensors
  String url = String(HF_SPACE_URL) + "/predict/compact";
  Serial.println("POST (CLOUD/HF) → " + url);
  http.begin(secureClient, url);
#elif USE_NGROK
  // ── HTTPS to ngrok tunnel ─────────────────────────────────────────
  secureClient.setBufferSizes(512, 512);
  secureClient.setInsecure();
  String url = String(NGROK_URL) + "/predict/compact";
  Serial.println("POST (NGROK) → " + url);
  http.begin(secureClient, url);
#else
  // ── HTTP to local Mac (fastest, no SSL overhead) ──────────────────
  String url = "http://" + String(SERVER_IP) + ":" + SERVER_PORT + "/predict/compact";
  Serial.println("POST (LOCAL) → " + url);
  http.begin(plainClient, url);
#endif

  http.addHeader("Content-Type", "application/json");
  http.addHeader("ngrok-skip-browser-warning", "true");  // skip ngrok warning page
  http.setTimeout(10000);

  code = http.POST(body);
  if (code == 200) {
    String resp = http.getString();
    Serial.println("Resp: " + resp);
    parseResponse(resp);
    g_serverOK = true;
  } else {
    Serial.printf("HTTP error: %d\n", code);
    g_serverOK = false;
    lcdStatus("Server Error", "HTTP " + String(code));
    delay(2000);
  }
  http.end();
}

// ══════════════════════════════════════════════════════════════════════
//   PARSE JSON RESPONSE FROM /predict/compact
// ══════════════════════════════════════════════════════════════════════
void parseResponse(String &body) {
  StaticJsonDocument<1024> doc;
  if (deserializeJson(doc, body)) {
    lcdStatus("Parse Error", "Bad JSON");
    delay(2000);
    return;
  }
  if (doc["ok"] == 0) {
    lcdStatus("Pred Error", doc["err"] | "unknown");
    delay(2000);
    return;
  }

  g_crop  = doc["crop"].as<String>();
  g_conf  = doc["conf"].as<float>();
  g_crop2 = doc["t2"].as<String>();
  g_conf2 = doc["c2"].as<float>();
  g_crop3 = doc["t3"].as<String>();
  g_conf3 = doc["c3"].as<float>();

  g_alerts     = doc["ac"].as<int>();
  g_alertCount = 0;

  JsonArray arr = doc["alerts"].as<JsonArray>();
  for (JsonObject a : arr) {
    if (g_alertCount >= 4) break;
    g_alertName[g_alertCount] = a["n"].as<String>();
    g_alertSev[g_alertCount]  = a["s"].as<String>();
    g_alertCount++;
  }

  // Reset alert page counter on new result
  g_alertPage = 0;

  Serial.printf("Crop:%s Conf:%.1f Alerts:%d\n",
                g_crop.c_str(), g_conf, g_alerts);
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN ROTATION LOGIC
// ══════════════════════════════════════════════════════════════════════
void advanceScreen() {
  /*
   * Screen 0: Sensor readings (always shown)
   * Screen 1: Crop recommendation
   * Screen 2: Disease alerts (only if g_alerts > 0, cycles through each)
   *
   * Cycle: 0 → 1 → (2 → 2 → ...) → 0 → ...
   */
  if (g_screen == 0) {
    g_screen = (g_crop.length() > 0) ? 1 : 0;
  } else if (g_screen == 1) {
    if (g_alerts > 0) {
      g_screen = 2;
      g_alertPage = 0;
    } else {
      g_screen = 0;
    }
  } else if (g_screen == 2) {
    g_alertPage++;
    if (g_alertPage >= g_alertCount) {
      g_screen = 0;
      g_alertPage = 0;
    }
    // Stay on screen 2 to cycle through each alert
  }
}

void drawScreen() {
  lcd.clear();
  switch (g_screen) {
    case 0: screenSensors(); break;
    case 1: screenCrop();    break;
    case 2: screenAlerts();  break;
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 0: SENSOR READINGS
// ══════════════════════════════════════════════════════════════════════
void screenSensors() {
  /*
   * Row 0: T:28.4^C H:72%
   * Row 1: Soil:45% R:WET
   *
   * ^ = custom degree character
   */
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(g_temp);   // int: no decimal places needed
  lcd.write(CHAR_DEG);           // degree symbol
  lcd.print("C H:");
  lcd.print((int)g_hum);
  lcd.print("%");

  lcd.setCursor(0, 1);
  lcd.print("Soil:");
  lcd.print(g_soil);
  lcd.print("% R:");
  if (g_rainOn) {
    lcd.write(CHAR_DROP);
    lcd.print("WET");
  } else {
    lcd.print("DRY");
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 1: CROP RECOMMENDATION
// ══════════════════════════════════════════════════════════════════════
void screenCrop() {
  if (g_crop.length() == 0) {
    lcd.setCursor(0, 0); lcd.print("No prediction");
    lcd.setCursor(0, 1); lcd.print("yet...");
    return;
  }

  /*
   * Row 0: [tick] PAPAYA
   * Row 1: Conf:62.1% 1ALT  (or alert warning)
   */
  String cropUp = g_crop;
  cropUp.toUpperCase();

  lcd.setCursor(0, 0);
  lcd.write(CHAR_TICK);
  lcd.print(" ");
  lcd.print(cropUp.substring(0, 13));   // max 13 chars (col 0-14)

  lcd.setCursor(0, 1);
  lcd.print("Conf:");
  lcd.print(g_conf, 1);
  lcd.print("% ");

  // Alert warning on right if any
  if (g_alerts > 0) {
    lcd.write(CHAR_ALERT);
    lcd.print(g_alerts);
    lcd.print("ALT");
  } else {
    lcd.write(CHAR_TICK);
    lcd.print("OK");
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 2: DISEASE ALERTS
// ══════════════════════════════════════════════════════════════════════
void screenAlerts() {
  if (g_alertCount == 0) {
    lcd.setCursor(3, 0);
    lcd.write(CHAR_TICK);
    lcd.print(" ALL CLEAR");
    lcd.setCursor(0, 1);
    lcd.print("No disease risk");
    return;
  }

  int i = g_alertPage;
  if (i >= g_alertCount) i = 0;

  /*
   * Row 0: !Anthracnose    (alert icon + name, truncated to 14)
   * Row 1: HIGH   Fungal   (severity + type indicator)
   */
  String name = g_alertName[i];
  String sev  = g_alertSev[i];

  lcd.setCursor(0, 0);
  lcd.write(CHAR_ALERT);
  lcd.print(name.substring(0, 14));   // 15 chars total (icon + 14 name chars)

  // Severity abbreviated
  lcd.setCursor(0, 1);
  if (sev == "CRITICAL") {
    lcd.print("!! CRITICAL !!");
  } else if (sev == "HIGH") {
    lcd.print("HIGH   ");
  } else if (sev == "MODERATE") {
    lcd.print("MODERATE");
  } else {
    lcd.print("WATCH  ");
  }

  // Page indicator if multiple alerts
  if (g_alertCount > 1) {
    lcd.setCursor(13, 1);
    lcd.print(i + 1);
    lcd.print("/");
    lcd.print(g_alertCount);
  }
}

// ══════════════════════════════════════════════════════════════════════
//   UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════════════════

// Show a two-line status message (used during transitions)
void lcdStatus(String line0, String line1) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line0.substring(0, 16));
  lcd.setCursor(0, 1); lcd.print(line1.substring(0, 16));
}

// ── Boot Screen ───────────────────────────────────────────────────────
void showBoot() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART PLANT AI");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  delay(1500);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");
  lcd.setCursor(0, 1);
  lcd.print(WIFI_SSID);
}

// ── WiFi Connection ───────────────────────────────────────────────────
void connectWiFi() {
  Serial.print("Connecting to: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("Connected! IP: ");
    Serial.println(WiFi.localIP());

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("WiFi Connected!");
    lcd.setCursor(0, 1);
    lcd.print(WiFi.localIP());
    delay(2500);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Server:");
    lcd.setCursor(0, 1);
#if USE_CLOUD
    lcd.print("HF Cloud");
#elif USE_NGROK
    lcd.print("ngrok PUBLIC");
#else
    lcd.print(SERVER_IP);
    lcd.print(":");
    lcd.print(SERVER_PORT);
#endif
    delay(2000);
  } else {
    Serial.println(F("WiFi FAILED!"));
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("!WiFi FAILED!");
    lcd.setCursor(0, 1);
    lcd.print("Check SSID/Pass");
    delay(10000);
  }
}
