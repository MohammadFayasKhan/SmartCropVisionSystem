/*
 * AgriRover is the embedded control firmware developed by Team Innovex for
 * the Smart Crop Intelligence and Crop Vision System proposed for SIH 2026.
 *
 * Team: Innovex
 * SIH Challenge ID: SH26180
 * Organization: Qualcomm Inc
 * Category: Hardware
 * Theme: Disaster Management
 *
 * The challenge focuses on a field-deployable AI-powered Smart Farming
 * Assistant that can help farmers identify crop diseases, pests, nutrient
 * deficiencies, and irrigation requirements at an early stage. It also
 * considers agricultural risks such as droughts, floods, and heat waves,
 * where timely information can help farmers respond more effectively.
 *
 * Our proposed solution combines a mobile rover, crop vision, field sensors,
 * and AI/ML analysis. The rover moves through crop rows and brings the
 * sensing system closer to the plants being inspected. The captured visual
 * and sensor information can then be passed to the intelligence layer for
 * detection, analysis, and decision support.
 *
 * This firmware is the real-time control layer underneath that system. It
 * runs on an ESP32 and handles rover movement, the field headlight, wireless
 * communication, browser-based teleoperation, device status, and the local
 * motor safety mechanism. Keeping these responsibilities separate from the
 * AI/ML inference layer gives the project a clear boundary between sensing,
 * intelligence, and physical actuation.
 *
 * Hardware used in this build:
 * ⤷ ESP32 DevKit
 * ⤷ L293D dual H-bridge motor driver
 * ⤷ Four TT DC gear motors
 * ⤷ External LED used as the rover headlight
 *
 * The four motors are arranged as two drive groups. The front and rear motors
 * on the left are connected in parallel to L293D channel M1. The front and
 * rear motors on the right are connected in parallel to channel M2.
 *
 * ESP32 to L293D connections:
 *   GPIO 5  → M1 IN1
 *   GPIO 18 → M1 IN2
 *   GPIO 19 → M2 IN3
 *   GPIO 21 → M2 IN4
 *   GPIO 2  → Rover headlight
 *
 * EN1 and EN2 are enabled with the driver board jumper caps.
 *
 * Network configuration:
 *   Station network : Fayas
 *   Fallback AP     : AgriRover
 *   AP password     : 12345678
 *   AP address      : 192.168.9.1
 *   mDNS address    : http://agrirover.local
 *
 * While a direction is being held, the browser sends a small heartbeat to
 * keep the movement command alive. The ESP32 stores the time of the latest
 * accepted command. If those heartbeats stop for longer than the configured
 * safety interval, the ESP32 stops both motor sides locally. The rover is
 * therefore not dependent on the browser remaining responsive for safe
 * operation.
 *
 * Because two TT motors are connected in parallel on each L293D channel,
 * startup and stall current can be much higher than normal running current.
 * The driver, battery, wiring, and connectors should be selected using the
 * measured motor requirements before the rover is used in the field.
 *
 * Target platform: Arduino-ESP32 3.3.x.
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// Firmware version identifies the software release installed on the rover.
constexpr const char *FIRMWARE_VERSION = "1.0.0";

// 1. Project and build configuration
//
// Keeping configuration in named constants makes the firmware easier to audit.
// It also prevents pin numbers and timing values from being scattered through
// motor, network, and HTTP code.
//

namespace Config {

  // Wi-Fi credentials used when the ESP32 tries to join the configured LAN.
  constexpr const char *STA_SSID = "Fayas";

  // Password for the configured station network.
  constexpr const char *STA_PASSWORD = "777888666";

  // SSID used when the ESP32 has to create its own control network.
  constexpr const char *AP_SSID = "AgriRover";

  // Password for the fallback access point.
  constexpr const char *AP_PASSWORD = "12345678";

  // Wi-Fi channel used by the fallback access point.
  constexpr uint8_t WIFI_CHANNEL = 6;

  // Maximum number of clients allowed on the fallback access point.
  constexpr uint8_t MAX_CLIENTS = 4;

  // Maximum time the rover may continue moving without a fresh command.
  constexpr uint32_t COMMAND_TIMEOUT_MS = 600;

  // Period between browser movement heartbeat requests.
  constexpr uint32_t HEARTBEAT_INTERVAL_MS = 250;

  // Period between browser status requests.
  constexpr uint32_t STATUS_POLL_INTERVAL_MS = 1000;

  // HTTP server port used by the local rover controller.
  constexpr uint16_t HTTP_PORT = 80;

  // Hostname advertised through mDNS.
  constexpr const char *MDNS_HOSTNAME = "agrirover";

  // Number of Wi-Fi connection attempts before the firmware falls back to AP.
  constexpr uint8_t WIFI_CONNECT_ATTEMPTS = 20;

  // Delay between Wi-Fi connection attempts.
  constexpr uint32_t WIFI_RETRY_DELAY_MS = 500;
}

// 2. Hardware pin configuration
//
// Each side of the rover uses one L293D channel. Both motors on a side receive
// the same two input signals.
//

namespace Pins {

  // L293D M1 input 1 for the two left motors.
  constexpr uint8_t LEFT_IN1 = 5;

  // L293D M1 input 2 for the two left motors.
  constexpr uint8_t LEFT_IN2 = 18;

  // L293D M2 input 1 for the two right motors.
  constexpr uint8_t RIGHT_IN1 = 19;

  // L293D M2 input 2 for the two right motors.
  constexpr uint8_t RIGHT_IN2 = 21;

  // ESP32 GPIO connected to the rover headlight or status LED.
  constexpr uint8_t STATUS_LED = 2;
}

// 3. Motor polarity and steering configuration
//
// The software direction depends on how each motor is mounted on
// the chassis. Keeping motor polarity here means the rest of the program can
// talk in terms of forward, reverse, left, and right without repeating those
// wiring details in every movement function.
//
// The current rover needs the pivot mapping inverted. With
// TURN_DIRECTION_INVERTED enabled, the LEFT command drives the left side
// forward and the right side in reverse. RIGHT does the opposite.

namespace MotorConfig {

   // Set true when the left motor pair is physically reversed relative to the
   // logical forward direction used by the rover software.
   constexpr bool LEFT_SIDE_INVERTED = false;

   // Set true when the right motor pair is physically reversed relative to the
   // logical forward direction used by the rover software.
   constexpr bool RIGHT_SIDE_INVERTED = false;

   // Select the pivot polarity required by the current rover chassis.
   constexpr bool TURN_DIRECTION_INVERTED = false;
 }

// 4. Server and runtime objects

// HTTP server instance that handles the rover control and status endpoints.
WebServer server(Config::HTTP_PORT);

// The embedded page is defined later in this file. This declaration lets the
// HTTP handler refer to it before the large flash-resident string is defined.
extern const char PAGE_INDEX[] PROGMEM;

// 5. Rover command model
//
// The enum gives the firmware one authoritative representation of movement.
// Using an enum instead of comparing strings prevents command spelling from
// leaking into motor-control logic.
//

enum class RoverCommand : uint8_t {

  // Both motor sides are stopped.
  STOPPED,

  // Both motor sides drive forward.
  FORWARD,

  // Both motor sides drive in reverse.
  BACKWARD,

  // Pivot command assigned to the left control action.
  LEFT,

  // Pivot command assigned to the right control action.
  RIGHT
};

// 6. Rover runtime state
//
// The state object contains only values that need to survive between HTTP
// requests or loop iterations.
//

struct RoverState {

  // Last movement command accepted by the firmware.
  RoverCommand currentCommand = RoverCommand::STOPPED;

  // Current state of the rover headlight.
  bool ledOn = false;

  // Time at which the most recent movement command or heartbeat was accepted.
  uint32_t lastCommandAt = 0;

  // Number of movement commands accepted since boot.
  uint32_t commandCount = 0;
};

// Single runtime state object used by the firmware.
RoverState state;

// Forward declaration for the watchdog helper used by movement handlers.
void touchWatchdog();

// 7. Utility functions

/**
 * The command enum is converted into readable text for the status API
 * and serial logs.
 */
const char *commandToString(RoverCommand command) {

  switch (command) {

    case RoverCommand::FORWARD:
      return "FORWARD";

    case RoverCommand::BACKWARD:
      return "BACKWARD";

    case RoverCommand::LEFT:
      return "LEFT";

    case RoverCommand::RIGHT:
      return "RIGHT";

    case RoverCommand::STOPPED:
    default:
      return "STOPPED";
  }
}

// 8. Low-level motor control
//
// The functions in this section are the only place where direction-to-pin
// polarity is translated into GPIO states. Keeping this boundary small makes
// the high-level movement code easier to inspect.
//

/**
 * The requested logical direction is translated into the GPIO state required
 * by one motor side. The side inversion is applied at this boundary, which
 * keeps the rest of the rover code expressed in physical movement terms.
 */
void writeMotorSide(
  uint8_t in1,
  uint8_t in2,
  bool inverted,
  bool forward
) {

  // A logical reverse request becomes a logical forward request when that
  // physical side has been wired with reversed motor polarity.
  if (inverted) {
    forward = !forward;
  }

  // L293D input pair for the selected side.
  if (forward) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  }
}

/**
 * The left L293D channel uses LOW/LOW as its inactive state, so both
 * inputs are driven LOW whenever the left drive group is stopped.
 */
void leftMotorStop() {

  digitalWrite(Pins::LEFT_IN1, LOW);
  digitalWrite(Pins::LEFT_IN2, LOW);
}

/**
 * The controller drives both left motors in the configured forward direction.
 */
void leftMotorForward() {

  writeMotorSide(
    Pins::LEFT_IN1,
    Pins::LEFT_IN2,
    MotorConfig::LEFT_SIDE_INVERTED,
    true
  );
}

/**
 * The controller drives both left motors in the configured reverse direction.
 */
void leftMotorReverse() {

  writeMotorSide(
    Pins::LEFT_IN1,
    Pins::LEFT_IN2,
    MotorConfig::LEFT_SIDE_INVERTED,
    false
  );
}

/**
 * The right L293D channel enters its inactive LOW/LOW input state.
 */
void rightMotorStop() {

  digitalWrite(Pins::RIGHT_IN1, LOW);
  digitalWrite(Pins::RIGHT_IN2, LOW);
}

/**
 * The controller drives both right motors in the configured forward direction.
 */
void rightMotorForward() {

  writeMotorSide(
    Pins::RIGHT_IN1,
    Pins::RIGHT_IN2,
    MotorConfig::RIGHT_SIDE_INVERTED,
    true
  );
}

/**
 * The controller drives both right motors in the configured reverse direction.
 */
void rightMotorReverse() {

  writeMotorSide(
    Pins::RIGHT_IN1,
    Pins::RIGHT_IN2,
    MotorConfig::RIGHT_SIDE_INVERTED,
    false
  );
}

// 9. Rover movement logic
//
// The rover uses skid-steer pivot commands. A turn drives the two sides in
// opposite directions instead of stopping one side.
//

/**
 * The controller forces every drive output into the stopped state and updates the software state
 * updated accordingly. The same function is shared by normal STOP requests and
 * the movement safety watchdog.
 */
void stopRover() {

  // Remove drive power from both sides before changing the reported state.
  leftMotorStop();
  rightMotorStop();

  // The firmware now considers the rover stationary.
  state.currentCommand = RoverCommand::STOPPED;
}

/**
 * The controller drives both sides forward together.
 */
void moveForward() {

  // Both drive sides move forward together for straight travel.
  leftMotorForward();
  rightMotorForward();

  state.currentCommand = RoverCommand::FORWARD;
}

/**
 * Both drive sides move in reverse together, matching the proven Rover.ino
 * four-wheel backward behavior.
 */
void moveBackward() {

  // Both drive sides reverse together.
  leftMotorReverse();
  rightMotorReverse();

  state.currentCommand = RoverCommand::BACKWARD;
}

/**
 * The LEFT command uses the same skid-steer mapping as Rover.ino.
 * The left side reverses while the right side moves forward.
 */
void turnLeft() {

  // Physical left pivot: left side reverse, right side forward.
  leftMotorReverse();
  rightMotorForward();

  state.currentCommand = RoverCommand::LEFT;
}

/**
 * The RIGHT command uses the same skid-steer mapping as Rover.ino.
 * The left side moves forward while the right side reverses.
 */
void turnRight() {

  // Physical right pivot: left side forward, right side reverse.
  leftMotorForward();
  rightMotorReverse();

  state.currentCommand = RoverCommand::RIGHT;
}

/**
 * The controller writes the requested headlight state to the GPIO output.
 */
void setLed(bool on) {

  // Store the requested state so /status and the web UI remain consistent.
  state.ledOn = on;

  // Apply the state to the physical LED pin.
  digitalWrite(
    Pins::STATUS_LED,
    on ? HIGH : LOW
  );
}

/**
 * The controller inverts the current headlight state through the normal LED
 * state handler.
 */
void toggleLed() {

  // Reuse setLed so the GPIO and state variable cannot drift apart.
  setLed(!state.ledOn);
}

// 11. Movement safety watchdog
//
// A network connection is not a safe movement signal by itself. The browser
// therefore refreshes the command timestamp while a button is held. If that
// stream stops, the ESP32 stops the motors locally.
//

/**
 * The controller records the latest accepted movement time as the local safety
 * deadline reference.
 */
void touchWatchdog() {

  // millis() is unsigned and subtraction remains safe across its rollover.
  state.lastCommandAt = millis();
}

/**
 * An expired movement heartbeat forces a local rover stop, independent of
 * the browser control page.
 */
void serviceWatchdog() {

  // There is nothing to supervise while the rover is already stopped.
  if (state.currentCommand == RoverCommand::STOPPED) {
    return;
  }

  // Compare elapsed time using unsigned subtraction so millis() rollover is
  // handled without a separate special case.
  const uint32_t elapsed =
    millis() - state.lastCommandAt;

  if (elapsed > Config::COMMAND_TIMEOUT_MS) {

    // Fail safe by stopping both motor channels before reporting the timeout.
    stopRover();

    Serial.println(
      F("[SAFETY] Movement heartbeat timeout. Rover stopped.")
    );
  }
}

// 12. Status API

/**
 * The status endpoint returns the current rover state as a compact JSON response. The endpoint
 * remains read-only, so it neither refreshes the watchdog nor changes motor
 * outputs.
 */
void sendJsonStatus() {

  // Use String here because WebServer accepts a String response directly and
  // the status payload is small enough for this application.
  String json;
  json.reserve(256);

  // Current movement command.
  json += F("{\"command\":\"");
  json += commandToString(state.currentCommand);
  json += F("\",");

  // Headlight state.
  json += F("\"led\":");
  json += state.ledOn ? F("true") : F("false");
  json += F(",");

  // RSSI is available when the ESP32 is operating as a station.
  if (WiFi.getMode() == WIFI_STA) {

    json += F("\"rssi\":");
    json += String(WiFi.RSSI());

  } else {

    // AP mode does not expose one meaningful upstream RSSI value.
    json += F("\"rssi\":null");
  }

  // Device uptime in seconds.
  json += F(",\"uptime\":");
  json += String(millis() / 1000);

  // Report the active interface address.
  json += F(",\"ip\":\"");

  if (WiFi.getMode() == WIFI_STA) {
    json += WiFi.localIP().toString();
  } else {
    json += WiFi.softAPIP().toString();
  }

  json += F("\"");

  // Number of stations connected to the ESP32 AP. In station mode this will
  // normally be zero.
  json += F(",\"clients\":");
  json += String(WiFi.softAPgetStationNum());

  // Firmware-side command counter.
  json += F(",\"commands\":");
  json += String(state.commandCount);

  // Close the JSON object.
  json += F("}");

  // Prevent browsers from displaying a stale status response.
  server.sendHeader(
    "Cache-Control",
    "no-cache, no-store, must-revalidate"
  );

  server.send(
    200,
    "application/json",
    json
  );
}

// 13. HTTP command handlers
//
// Every movement handler updates the watchdog timestamp only after the command
// has been accepted and the motor outputs have been written.
//

/**
 * The ESP32 serves the embedded rover controller directly from program memory.
 */
void handleRoot() {

  server.sendHeader(
    "Cache-Control",
    "no-cache, no-store, must-revalidate"
  );

  server.send_P(
    200,
    "text/html",
    PAGE_INDEX
  );
}

/**
 * The handler converts a forward request into the rover's forward state.
 */
void handleForward() {

  moveForward();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler converts a backward request into the rover's reverse state.
 */
void handleBackward() {

  moveBackward();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler converts the left control request into the configured physical left pivot.
 */
void handleLeft() {

  turnLeft();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler converts the right control request into the configured physical right pivot.
 */
void handleRight() {

  turnRight();
  touchWatchdog();
  state.commandCount++;

  server.send(200, "text/plain", "OK");
}

/**
 * The handler stops the rover immediately.
 */
void handleStop() {

  stopRover();
  touchWatchdog();

  server.send(200, "text/plain", "OK");
}

/**
 * The handler switches the headlight ON.
 */
void handleLedOn() {

  setLed(true);

  server.send(200, "text/plain", "OK");
}

/**
 * The handler switches the headlight OFF.
 */
void handleLedOff() {

  setLed(false);

  server.send(200, "text/plain", "OK");
}

/**
 * The handler toggles the current headlight state.
 */
void handleLedToggle() {

  toggleLed();

  server.send(200, "text/plain", "OK");
}

/**
 * The handler returns the current device state to the browser.
 */
void handleStatus() {
  sendJsonStatus();
}

/**
 * The server returns a controlled 404 response for unknown HTTP paths.
 */
void handleNotFound() {

  server.send(
    404,
    "text/plain",
    "404: Route not found"
  );
}

// 14. HTTP route registration

/**
 * The server registers the movement, headlight, status, and fallback HTTP routes.
 */
void registerRoutes() {

  // Main controller page.
  server.on("/", HTTP_GET, handleRoot);

  // Movement endpoints.
  server.on("/forward", HTTP_GET, handleForward);
  server.on("/backward", HTTP_GET, handleBackward);
  server.on("/left", HTTP_GET, handleLeft);
  server.on("/right", HTTP_GET, handleRight);
  server.on("/stop", HTTP_GET, handleStop);

  // Headlight endpoints.
  server.on("/led/on", HTTP_GET, handleLedOn);
  server.on("/led/off", HTTP_GET, handleLedOff);
  server.on("/led/toggle", HTTP_GET, handleLedToggle);

  // Read-only device status.
  server.on("/status", HTTP_GET, handleStatus);

  // All unmatched requests receive a deterministic 404 response.
  server.onNotFound(handleNotFound);
}

// 15. Embedded web application
//
// The page is stored in flash through PROGMEM so the large HTML/CSS/JavaScript
// payload does not consume the same RAM used by the rover runtime.
//

const char PAGE_INDEX[] PROGMEM = R"HTML_PAGE(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">

<meta
  name="viewport"
  content="width=device-width,initial-scale=1.0,maximum-scale=1.0,minimum-scale=1.0,user-scalable=no,viewport-fit=cover"
>

<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0b1710">

<title>AgriRover | Smart Crop Intelligence</title>

<style>
:root {
  --bg:#0b1710;
  --surface:#101f15;
  --surface2:#14271a;
  --border:#294532;
  --green:#5fcf72;
  --green-light:#8ee69b;
  --green-dark:#2f8f45;
  --leaf:#76c893;
  --soil:#b58b63;
  --yellow:#e5c95d;
  --red:#ef6461;
  --text:#f4f8f3;
  --text2:#c8d6c9;
  --text3:#849688;
  --sans:Inter,-apple-system,BlinkMacSystemFont,"SF Pro Display","Helvetica Neue",Arial,sans-serif;
  --mono:"SF Mono",Menlo,Consolas,monospace;
  --radius:18px;
  --radius-lg:24px;
}

html,
body {
  width:100%;
  height:100%;
  margin:0;
  padding:0;
  overflow:hidden;
  position:fixed;
  inset:0;
  touch-action:none;
  overscroll-behavior:none;
  -webkit-text-size-adjust:100%;
  text-size-adjust:100%;
  background:
    radial-gradient(circle at 20% 0%,rgba(82,145,85,.14),transparent 35%),
    radial-gradient(circle at 100% 100%,rgba(181,139,99,.08),transparent 35%),
    var(--bg);
  color:var(--text);
  font-family:var(--sans);
  -webkit-font-smoothing:antialiased;
}

* {
  box-sizing:border-box;
  margin:0;
  padding:0;
  -webkit-tap-highlight-color:transparent;
  user-select:none;
  -webkit-user-select:none;
  -webkit-touch-callout:none;
}

button {
  font-family:inherit;
  touch-action:none;
  user-select:none;
  -webkit-user-select:none;
  -webkit-touch-callout:none;
}

.app {
  position:fixed;
  inset:0;
  width:100vw;
  height:100vh;
  height:100dvh;
  display:grid;
  grid-template-rows:auto auto auto 1fr auto;
  padding:
    env(safe-area-inset-top,10px)
    calc(env(safe-area-inset-right,0px) + 16px)
    calc(env(safe-area-inset-bottom,8px) + 6px)
    calc(env(safe-area-inset-left,0px) + 16px);
  gap:8px;
  overflow:hidden;
}

header {
  display:flex;
  align-items:center;
  justify-content:space-between;
  background:linear-gradient(135deg,rgba(31,57,36,.96),rgba(13,30,18,.96));
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  padding:11px 14px;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.025),0 10px 30px rgba(0,0,0,.25);
}

.brand {
  display:flex;
  align-items:center;
  gap:10px;
}

.logo-box {
  width:42px;
  height:42px;
  border-radius:13px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:linear-gradient(145deg,#285c35,#173a22);
  border:1px solid rgba(118,200,147,.35);
}

.logo {
  width:26px;
  height:26px;
}

.brand h1 {
  font-size:18px;
  font-weight:800;
  letter-spacing:-.025em;
}

.brand p {
  margin-top:2px;
  font-size:8px;
  color:var(--text3);
  font-weight:700;
  letter-spacing:.07em;
  text-transform:uppercase;
}

.link-pill {
  display:flex;
  align-items:center;
  gap:6px;
  padding:7px 10px;
  border-radius:20px;
  background:rgba(0,0,0,.16);
  border:1px solid var(--border);
  color:var(--text3);
  font-size:9px;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.05em;
}

.dot {
  width:7px;
  height:7px;
  border-radius:50%;
  background:var(--text3);
}

.dot.live {
  background:var(--green);
  box-shadow:0 0 10px rgba(95,207,114,.8);
  animation:pulse 1.5s infinite;
}

.dot.dead {
  background:var(--red);
  box-shadow:0 0 10px rgba(239,100,97,.65);
}

@keyframes pulse {
  0%,100% { opacity:1; }
  50% { opacity:.35; }
}

.stats {
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:6px;
}

.stat {
  background:linear-gradient(145deg,rgba(24,45,29,.98),rgba(13,29,18,.98));
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:8px 4px;
  text-align:center;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 7px 18px rgba(0,0,0,.2);
}

.stat .lbl {
  color:var(--text3);
  font-size:7px;
  font-weight:800;
  letter-spacing:.06em;
  text-transform:uppercase;
}

.stat .val {
  margin-top:2px;
  color:var(--text);
  font-size:11px;
  font-weight:800;
  font-variant-numeric:tabular-nums;
}

.stat .go { color:var(--green-light); }
.stat .warn { color:var(--yellow); }
.stat .err { color:var(--red); }

.led-row {
  display:flex;
  align-items:center;
  gap:11px;
  width:100%;
  padding:9px 12px;
  background:linear-gradient(145deg,rgba(25,48,30,.96),rgba(13,29,18,.96));
  border:1px solid var(--border);
  border-radius:var(--radius);
  cursor:pointer;
  touch-action:manipulation;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 7px 20px rgba(0,0,0,.2);
}

.led-row.on {
  border-color:rgba(95,207,114,.4);
  box-shadow:inset 0 0 18px rgba(95,207,114,.06),0 0 20px rgba(95,207,114,.1);
}

.led-icon {
  width:34px;
  height:34px;
  display:flex;
  align-items:center;
  justify-content:center;
  border-radius:11px;
  background:rgba(95,207,114,.08);
  font-size:18px;
  filter:grayscale(1);
  opacity:.6;
}

.led-row.on .led-icon {
  filter:none;
  opacity:1;
}

.led-info {
  flex:1;
}

.led-name {
  font-size:12px;
  font-weight:800;
}

.led-description {
  margin-top:1px;
  font-size:8px;
  color:var(--text3);
  text-transform:uppercase;
  letter-spacing:.04em;
}

.toggle {
  position:relative;
  width:43px;
  height:25px;
  border-radius:20px;
  background:rgba(255,255,255,.09);
}

.toggle.on {
  background:var(--green-dark);
  box-shadow:0 0 10px rgba(95,207,114,.3);
}

.toggle-thumb {
  position:absolute;
  left:2px;
  top:2px;
  width:21px;
  height:21px;
  border-radius:50%;
  background:white;
  box-shadow:0 2px 5px rgba(0,0,0,.3);
  transition:left .18s cubic-bezier(.16,1,.3,1);
}

.toggle.on .toggle-thumb {
  left:20px;
}

main {
  min-height:0;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:10px;
  overflow:hidden;
}

.log-box {
  width:100%;
  height:96px;
  padding:9px 11px;
  background:rgba(10,22,14,.92);
  border:1px solid var(--border);
  border-radius:var(--radius);
  overflow:hidden;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 8px 20px rgba(0,0,0,.2);
  font-family:var(--mono);
  font-size:8px;
}

.log-header {
  display:flex;
  justify-content:space-between;
  padding-bottom:5px;
  margin-bottom:4px;
  border-bottom:1px solid rgba(118,200,147,.1);
  color:var(--text3);
  font-size:7px;
  font-weight:800;
  letter-spacing:.06em;
  text-transform:uppercase;
}

.log-body {
  display:flex;
  flex-direction:column-reverse;
  height:calc(100% - 20px);
  overflow:hidden;
}

.log-line {
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  opacity:0;
  animation:lineIn .2s ease forwards;
}

@keyframes lineIn {
  from { opacity:0; transform:translateY(3px); }
  to { opacity:1; transform:none; }
}

.ts { color:#627265; margin-right:5px; }
.c-fwd { color:var(--green-light); font-weight:700; }
.c-back { color:var(--soil); font-weight:700; }
.c-left,.c-right { color:var(--leaf); font-weight:700; }
.c-stop { color:var(--red); font-weight:700; }
.c-led { color:var(--yellow); font-weight:700; }

.ctrl-area {
  position:relative;
  display:flex;
  align-items:center;
  justify-content:center;
  margin:auto 0;
}

.ctrl-ring {
  position:absolute;
  width:min(55vw,215px);
  height:min(55vw,215px);
  border-radius:50%;
  border:2px dashed rgba(95,207,114,.18);
  pointer-events:none;
  box-shadow:inset 0 0 25px rgba(95,207,114,.035);
}

.ctrl-ring.on {
  border-color:rgba(95,207,114,.6);
  box-shadow:inset 0 0 30px rgba(95,207,114,.12),0 0 25px rgba(95,207,114,.12);
  animation:rotateRing 16s linear infinite;
}

@keyframes rotateRing {
  to { transform:rotate(360deg); }
}

.dpad {
  position:relative;
  width:min(48vw,185px);
  height:min(48vw,185px);
  display:grid;
  grid-template-columns:repeat(3,1fr);
  grid-template-rows:repeat(3,1fr);
  gap:5px;
}

.dir {
  appearance:none;
  border:1px solid var(--border);
  outline:none;
  background:linear-gradient(145deg,rgba(26,49,31,.98),rgba(13,29,18,.98));
  color:var(--text2);
  display:flex;
  align-items:center;
  justify-content:center;
  cursor:pointer;
  touch-action:none;
  transition:all .12s cubic-bezier(.16,1,.3,1);
  box-shadow:0 5px 13px rgba(0,0,0,.25);
}

.dir-up {
  grid-column:2;
  grid-row:1;
  border-radius:16px 16px 8px 8px;
}

.dir-left {
  grid-column:1;
  grid-row:2;
  border-radius:16px 8px 8px 16px;
}

.dir-right {
  grid-column:3;
  grid-row:2;
  border-radius:8px 16px 16px 8px;
}

.dir-down {
  grid-column:2;
  grid-row:3;
  border-radius:8px 8px 16px 16px;
}

.dir:active,
.dir.active {
  transform:scale(.92);
  color:var(--green-light);
  background:rgba(95,207,114,.15);
  border-color:rgba(95,207,114,.5);
  box-shadow:inset 0 0 16px rgba(95,207,114,.15),0 0 15px rgba(95,207,114,.15);
}

.dir svg {
  width:21px;
  height:21px;
  stroke:currentColor;
  fill:none;
  stroke-width:2;
  stroke-linecap:round;
  stroke-linejoin:round;
}

.stop-btn {
  grid-column:2;
  grid-row:2;
  border-radius:50%;
  appearance:none;
  outline:none;
  cursor:pointer;
  touch-action:none;
  border:1px solid rgba(239,100,97,.55);
  background:linear-gradient(145deg,#a83f3d,#702c2b);
  color:white;
  font-size:10px;
  font-weight:900;
  letter-spacing:.04em;
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow:0 0 16px rgba(239,100,97,.18);
}

.stop-btn:active,
.stop-btn.active {
  transform:scale(.9);
  box-shadow:0 0 25px rgba(239,100,97,.45);
}

.footer {
  width:fit-content;
  margin:0 auto;
  padding:6px 12px;
  border-radius:18px;
  background:rgba(16,31,21,.9);
  border:1px solid var(--border);
  color:var(--text3);
  font-size:7px;
  font-weight:800;
  letter-spacing:.045em;
  text-transform:uppercase;
  text-align:center;
}

.footer span {
  color:var(--green-light);
}

#boot {
  position:fixed;
  inset:0;
  z-index:50;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:7px;
  background:
    radial-gradient(circle at center,rgba(50,105,57,.18),transparent 45%),
    var(--bg);
  transition:opacity .5s ease,visibility .5s ease;
}

#boot.hide {
  opacity:0;
  visibility:hidden;
  pointer-events:none;
}

.boot-logo {
  width:68px;
  height:68px;
  display:flex;
  align-items:center;
  justify-content:center;
  border-radius:20px;
  background:linear-gradient(145deg,#326d3f,#183a22);
  border:1px solid rgba(118,200,147,.4);
  box-shadow:0 0 35px rgba(95,207,114,.2);
}

.boot-title {
  font-size:20px;
  font-weight:900;
}

.boot-sub {
  font-size:10px;
  color:var(--text3);
  font-weight:600;
}

.boot-bar {
  width:150px;
  height:3px;
  margin-top:10px;
  border-radius:5px;
  background:rgba(255,255,255,.06);
  overflow:hidden;
}

.boot-fill {
  height:100%;
  width:0;
  border-radius:5px;
  background:linear-gradient(90deg,var(--green-dark),var(--green-light));
  animation:bootLoad .9s ease forwards;
}

@keyframes bootLoad {
  to { width:100%; }
}

@media (orientation:landscape) and (max-height:480px) {
  .app {
    grid-template-rows:auto auto 1fr auto;
    grid-template-columns:1fr 1fr;
  }

  header {
    grid-column:1 / 3;
  }

  .stats {
    grid-column:1 / 3;
    grid-template-columns:repeat(6,1fr);
  }

  .led-row {
    display:none;
  }

  main {
    grid-column:1 / 3;
    flex-direction:row;
    justify-content:space-evenly;
  }

  .log-box {
    width:220px;
    height:100%;
  }

  .footer {
    display:none;
  }
}
</style>
</head>

<body>

<div id="boot">
  <div class="boot-logo">
    <svg width="38" height="38" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path d="M38 8C23 9 13 15 11 27C10 34 14 39 20 40C29 41 38 30 38 8Z" stroke="#ffffff" stroke-width="2.5" stroke-linejoin="round"/>
      <path d="M12 36C18 28 24 22 35 14" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
      <rect x="8" y="31" width="32" height="9" rx="3" stroke="#ffffff" stroke-width="2"/>
      <circle cx="15" cy="43" r="3" stroke="#ffffff" stroke-width="2"/>
      <circle cx="33" cy="43" r="3" stroke="#ffffff" stroke-width="2"/>
    </svg>
  </div>

  <div class="boot-title">AgriRover</div>

  <div class="boot-sub" id="bootLine">
    Initializing crop intelligence rover...
  </div>

  <div class="boot-bar">
    <div class="boot-fill"></div>
  </div>
</div>

<div class="app">

  <header>
    <div class="brand">
      <div class="logo-box">
        <svg class="logo" viewBox="0 0 48 48" fill="none" aria-hidden="true">
          <path d="M37 8C24 9 14 15 12 26C11 33 14 38 20 39C29 40 37 29 37 8Z" stroke="#8ee69b" stroke-width="2.5" stroke-linejoin="round"/>
          <path d="M12 35C19 27 25 21 34 14" stroke="#8ee69b" stroke-width="2" stroke-linecap="round"/>
          <rect x="8" y="31" width="32" height="8" rx="3" stroke="#8ee69b" stroke-width="2"/>
          <circle cx="15" cy="42" r="2.5" fill="#8ee69b"/>
          <circle cx="33" cy="42" r="2.5" fill="#8ee69b"/>
        </svg>
      </div>

      <div>
        <h1>AgriRover</h1>
        <p>Smart Crop Intelligence</p>
      </div>
    </div>

    <div class="link-pill">
      <span class="dot" id="connDot"></span>
      <span id="connText">Link</span>
    </div>
  </header>

  <div class="stats">

    <div class="stat">
      <div class="lbl">Rover</div>
      <div class="val" id="stCommand">Idle</div>
    </div>

    <div class="stat">
      <div class="lbl">Light</div>
      <div class="val err" id="stLed">Off</div>
    </div>

    <div class="stat">
      <div class="lbl">Signal</div>
      <div class="val" id="stRssi">--</div>
    </div>

    <div class="stat">
      <div class="lbl">Ping</div>
      <div class="val" id="stPing">--</div>
    </div>

    <div class="stat">
      <div class="lbl">Uptime</div>
      <div class="val" id="stUptime">0s</div>
    </div>

    <div class="stat">
      <div class="lbl">Commands</div>
      <div class="val" id="stCount">0</div>
    </div>

  </div>

  <div class="led-row" id="ledToggle">
    <div class="led-icon" aria-hidden="true">💡</div>

    <div class="led-info">
      <div class="led-name">Field Headlight</div>

      <div class="led-description" id="ledSub">
        Tap to illuminate crop rows
      </div>
    </div>

    <div class="toggle" id="ledSwitch" aria-hidden="true">
      <div class="toggle-thumb"></div>
    </div>
  </div>

  <main>

    <div class="log-box">
      <div class="log-header">
        <span>Rover Activity</span>
        <span id="termClock">00:00:00</span>
      </div>

      <div class="log-body" id="termBody"></div>
    </div>

    <div class="ctrl-area">
      <div class="ctrl-ring" id="dpadRing"></div>

      <div class="dpad" id="dpad">

        <button class="dir dir-up" data-cmd="forward" aria-label="Forward">
          <svg viewBox="0 0 24 24">
            <polyline points="6 15 12 9 18 15"/>
          </svg>
        </button>

        <button class="dir dir-left" data-cmd="left" aria-label="Left">
          <svg viewBox="0 0 24 24">
            <polyline points="15 18 9 12 15 6"/>
          </svg>
        </button>

        <button class="stop-btn" data-cmd="stop" aria-label="Stop">
          STOP
        </button>

        <button class="dir dir-right" data-cmd="right" aria-label="Right">
          <svg viewBox="0 0 24 24">
            <polyline points="9 6 15 12 9 18"/>
          </svg>
        </button>

        <button class="dir dir-down" data-cmd="backward" aria-label="Backward">
          <svg viewBox="0 0 24 24">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>

      </div>
    </div>
  </main>

  <div class="footer">
    Smart Crop Intelligence + Crop Vision System
    &nbsp;•&nbsp;
    Team <span>Innovex</span>
  </div>

</div>

<script>
(() => {
  "use strict";

  // DOM references used by the controller.
  const $ = id => document.getElementById(id);

  const dpad = $("dpad");
  const dpadRing = $("dpadRing");
  const connDot = $("connDot");
  const connText = $("connText");
  const stCommand = $("stCommand");
  const stLed = $("stLed");
  const stRssi = $("stRssi");
  const stPing = $("stPing");
  const stUptime = $("stUptime");
  const stCount = $("stCount");
  const ledToggle = $("ledToggle");
  const ledSwitch = $("ledSwitch");
  const ledSub = $("ledSub");
  const boot = $("boot");
  const bootLine = $("bootLine");
  const termBody = $("termBody");
  const termClock = $("termClock");

  // HTTP request timeout prevents a stalled browser request from hanging the
  // controller UI indefinitely.
  const REQ_TIMEOUT = 1200;

  // This interval stays below the ESP32 safety timeout.
  const HEARTBEAT = 250;

  // Status polling is independent of movement heartbeats.
  const POLL_MS = 1000;

  // The activity log is deliberately small so the DOM remains cheap on mobile.
  const MAX_LOG = 7;

  // Current movement command being held by a pointer or keyboard.
  let activeCmd = null;

  // Browser timer used for the movement heartbeat.
  let hbTimer = null;

  // Last known transport state.
  let connected = false;

  // Local UI count of movement actions.
  let cmdCount = 0;

  /**
   * The controller triggers short device vibration when the browser exposes haptic feedback.
   */
  function haptic(type) {

    try {
      if (!navigator.vibrate) {
        return;
      }

      const patterns = {
        light: [8],
        medium: [15],
        heavy: [25],
        error: [15, 40, 15],
        success: [10, 30, 10, 30, 10]
      };

      navigator.vibrate(
        patterns[type] || patterns.light
      );

    } catch (error) {
      // Vibration is optional. A browser error must not affect movement.
    }
  }

  // Boot messages give the user feedback while the page becomes interactive.
  const bootMessages = [
    "Starting AgriRover core...",
    "Initializing four-wheel motor control...",
    "Establishing field network...",
    "Crop intelligence link ready."
  ];

  // Index of the boot message currently displayed.
  let bootIndex = 0;

  // Rotate through the short boot sequence.
  const bootInterval = setInterval(() => {

    bootIndex++;

    if (bootIndex < bootMessages.length) {
      bootLine.textContent = bootMessages[bootIndex];
    } else {
      clearInterval(bootInterval);
    }

  }, 250);

  // Browser-side time used only for activity-log timestamps.
  const startTime = Date.now();

  // CSS classes used to distinguish activity entries.
  const tagMap = {
    forward: "c-fwd",
    backward: "c-back",
    left: "c-left",
    right: "c-right",
    stop: "c-stop",
    led: "c-led"
  };

  /**
   * The controller adds each selected event to the compact activity log.
   */
  function log(text, cls) {

    const element = document.createElement("div");

    element.className = "log-line";

    const seconds =
      ((Date.now() - startTime) / 1000).toFixed(1);

    // The text is generated locally from fixed command labels. No user input
    // is inserted into innerHTML.
    element.innerHTML =
      '<span class="ts">+' +
      seconds +
      's</span>' +
      '<span class="' +
      (cls || "") +
      '">' +
      text +
      '</span>';

    termBody.insertBefore(
      element,
      termBody.firstChild
    );

    // Remove the oldest entries once the configured display size is reached.
    while (termBody.children.length > MAX_LOG) {
      termBody.removeChild(termBody.lastChild);
    }
  }

  log("AgriRover system online", "c-led");

  /**
   * The controller refreshes the activity clock from the browser's local time.
   */
  function updateClock() {

    const now = new Date();

    termClock.textContent = [
      now.getHours(),
      now.getMinutes(),
      now.getSeconds()
    ]
      .map(n => String(n).padStart(2, "0"))
      .join(":");
  }

  setInterval(updateClock, 1000);
  updateClock();

  /**
   * Update the connection indicator without creating repeated log entries for
   * the same connection state.
   */
  function markConnection(ok) {

    if (ok === connected) {
      return;
    }

    connected = ok;

    connDot.className =
      "dot " +
      (ok ? "live" : "dead");

    connText.textContent =
      ok ? "Linked" : "Offline";

    if (!ok) {
      log(
        "Rover connection lost",
        "c-stop"
      );
    }
  }

  /**
   * The controller sends a bounded-timeout GET request to an ESP32 control endpoint.
   */
  function request(path) {

    const controller = new AbortController();
    const start = performance.now();

    const timeout = setTimeout(
      () => controller.abort(),
      REQ_TIMEOUT
    );

    return fetch(
      path,
      {
        method: "GET",
        signal: controller.signal,
        cache: "no-store"
      }
    )
      .then(response => {

        clearTimeout(timeout);

        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }

        markConnection(true);

        stPing.textContent =
          Math.round(
            performance.now() - start
          ) + " ms";

        return response;
      })
      .catch(error => {

        clearTimeout(timeout);
        markConnection(false);

        throw error;
      });
  }

  /**
   * The controller updates the visible movement state immediately after issuing a command.
   */
  function updateMovementUI(cmd) {

    const moving = cmd !== "stop";
    const label = cmd.toUpperCase();

    stCommand.textContent =
      label === "STOP"
        ? "Idle"
        : label;

    stCommand.className =
      "val" +
      (moving ? " go" : "");

    dpadRing.classList.toggle(
      "on",
      moving
    );
  }

  /**
   * Send a movement or stop command to the ESP32.
   *
   * Heartbeats intentionally skip the activity log and command counter so a
   * held button does not flood the UI with duplicate entries.
   */
  function send(cmd, heartbeat = false) {

    request("/" + cmd)
      .catch(() => {});

    updateMovementUI(cmd);

    if (!heartbeat) {

      cmdCount++;
      stCount.textContent = cmdCount;

      log(
        cmd === "stop"
          ? "Rover stopped"
          : cmd.toUpperCase() + " engaged",
        tagMap[cmd]
      );

      haptic(
        cmd === "stop"
          ? "light"
          : "medium"
      );
    }
  }

  /**
   * The controller starts the held movement command and maintains its watchdog heartbeat.
   */
  function holdStart(cmd) {

    if (activeCmd === cmd) {
      return;
    }

    // Stop the previous command before changing direction. This avoids leaving
    // a previous heartbeat active while another direction is selected.
    if (activeCmd) {

      clearInterval(hbTimer);
      activeCmd = null;

      send("stop", true);
    }

    activeCmd = cmd;

    // Send immediately rather than waiting for the first heartbeat interval.
    send(cmd, false);

    clearInterval(hbTimer);

    hbTimer = setInterval(() => {

      if (activeCmd) {
        send(activeCmd, true);
      }

    }, HEARTBEAT);
  }

  /**
   * The controller releases the held movement command and sends the final stop request.
   */
  function holdEnd() {

    if (!activeCmd) {
      return;
    }

    activeCmd = null;

    clearInterval(hbTimer);
    hbTimer = null;

    send("stop", false);
  }

  /**
   * The controller attaches pointer events to the direction and stop controls.
   */
  dpad
    .querySelectorAll("button")
    .forEach(button => {

      const cmd = button.dataset.cmd;

      button.addEventListener(
        "pointerdown",
        event => {

          event.preventDefault();

          try {
            button.setPointerCapture(event.pointerId);
          } catch (error) {
            // Pointer capture is a convenience. Movement still works without it.
          }

          button.classList.add("active");

          if (cmd === "stop") {

            activeCmd = null;

            clearInterval(hbTimer);
            hbTimer = null;

            send("stop", false);
            haptic("error");

          } else {

            holdStart(cmd);
          }
        },
        { passive: false }
      );

      button.addEventListener(
        "pointerup",
        event => {

          event.preventDefault();
          button.classList.remove("active");

          if (cmd !== "stop") {
            holdEnd();
          }
        },
        { passive: false }
      );

      button.addEventListener(
        "pointercancel",
        event => {

          event.preventDefault();
          button.classList.remove("active");

          if (cmd !== "stop") {
            holdEnd();
          }
        },
        { passive: false }
      );

      button.addEventListener(
        "lostpointercapture",
        () => {
          button.classList.remove("active");
        }
      );

      button.addEventListener(
        "contextmenu",
        event => event.preventDefault()
      );
    });

  // Keyboard movement mapping for desktop testing and field laptop control.
  const keyboardMap = {
    w: "forward",
    arrowup: "forward",
    s: "backward",
    arrowdown: "backward",
    a: "left",
    arrowleft: "left",
    d: "right",
    arrowright: "right"
  };

  // Map command names to their corresponding UI buttons.
  const commandButtons = {};

  dpad
    .querySelectorAll("button")
    .forEach(button => {
      commandButtons[button.dataset.cmd] = button;
    });

  /**
   * The controller maps keyboard input to the same movement commands used by the touch controls.
   */
  window.addEventListener(
    "keydown",
    event => {

      const key = event.key.toLowerCase();

      // Space always requests an immediate stop.
      if (key === " ") {

        event.preventDefault();

        send("stop", false);

        activeCmd = null;

        clearInterval(hbTimer);
        hbTimer = null;

        haptic("error");

        return;
      }

      const cmd = keyboardMap[key];

      if (!cmd || event.repeat) {
        return;
      }

      event.preventDefault();

      holdStart(cmd);

      if (commandButtons[cmd]) {
        commandButtons[cmd].classList.add("active");
      }
    }
  );

  /**
   * Keyboard release events terminate the active movement command.
   */
  window.addEventListener(
    "keyup",
    event => {

      const cmd =
        keyboardMap[
          event.key.toLowerCase()
        ];

      if (!cmd) {
        return;
      }

      event.preventDefault();

      holdEnd();

      if (commandButtons[cmd]) {
        commandButtons[cmd].classList.remove("active");
      }
    }
  );

  /**
   * The controller synchronizes every visible headlight control with the current state.
   */
  function setLedUI(on) {

    ledSwitch.classList.toggle("on", on);
    ledToggle.classList.toggle("on", on);

    stLed.textContent =
      on ? "On" : "Off";

    stLed.className =
      "val " +
      (on ? "warn" : "err");

    ledSub.textContent =
      on
        ? "Tap to switch off"
        : "Tap to illuminate crop rows";
  }

  // Headlight row acts as one large touch target.
  ledToggle.addEventListener(
    "click",
    () => {

      const on =
        !ledSwitch.classList.contains("on");

      request(
        "/led/" +
        (on ? "on" : "off")
      )
        .catch(() => {});

      setLedUI(on);

      haptic("light");

      log(
        "Field headlight " +
        (on ? "ON" : "OFF"),
        "c-led"
      );
    }
  );

  /**
   * Request a stop without waiting for normal movement state handling.
   *
   * GET is used because the ESP32 exposes /stop as HTTP_GET. sendBeacon() is
   * intentionally avoided because browsers normally send it as POST.
   */
  function emergencyStop() {

    activeCmd = null;

    clearInterval(hbTimer);
    hbTimer = null;

    try {

      fetch(
        "/stop",
        {
          method: "GET",
          cache: "no-store",
          keepalive: true
        }
      )
        .catch(() => {});

    } catch (error) {
      // The local watchdog remains the final safety layer if the browser fails.
    }
  }

  // Stop the rover when the page is being discarded or hidden.
  window.addEventListener("beforeunload", emergencyStop);
  window.addEventListener("pagehide", emergencyStop);

  document.addEventListener(
    "visibilitychange",
    () => {
      if (document.hidden) {
        emergencyStop();
      }
    }
  );

  // Prevent browser gesture zoom on supported mobile browsers.
  [
    "gesturestart",
    "gesturechange",
    "gestureend"
  ]
    .forEach(type => {

      document.addEventListener(
        type,
        event => event.preventDefault(),
        { passive: false }
      );
    });

  // Prevent touch scrolling inside the controller surface.
  document.addEventListener(
    "touchmove",
    event => event.preventDefault(),
    { passive: false }
  );

  // Prevent Ctrl + wheel browser zoom on desktop browsers.
  document.addEventListener(
    "wheel",
    event => {
      if (event.ctrlKey) {
        event.preventDefault();
      }
    },
    { passive: false }
  );

  // Prevent double-tap zoom where the browser implements that gesture.
  let lastTouchEnd = 0;

  document.addEventListener(
    "touchend",
    event => {

      const now = Date.now();

      if (now - lastTouchEnd <= 300) {
        event.preventDefault();
      }

      lastTouchEnd = now;
    },
    { passive: false }
  );

  /**
   * The browser periodically reads the ESP32's read-only device state and refreshes the interface.
   */
  function pollStatus() {

    request("/status")
      .then(response => response.json())
      .then(data => {

        const label =
          data.command === "STOPPED"
            ? "Idle"
            : data.command;

        stCommand.textContent = label;

        const moving =
          data.command !== "STOPPED";

        stCommand.className =
          "val" +
          (moving ? " go" : "");

        dpadRing.classList.toggle(
          "on",
          moving
        );

        if (data.rssi !== null) {
          stRssi.textContent =
            data.rssi + " dBm";
        } else {
          stRssi.textContent = "AP";
        }

        stUptime.textContent =
          formatUptime(data.uptime);

        if (typeof data.commands === "number") {
          stCount.textContent = data.commands;
        }

        setLedUI(data.led);
      })
      .catch(() => {});
  }

  /**
   * The controller formats uptime seconds into a compact hours, minutes, and seconds display.
   */
  function formatUptime(seconds) {

    if (seconds < 60) {
      return seconds + "s";
    }

    const minutes =
      Math.floor(seconds / 60);

    const remainder =
      seconds % 60;

    if (minutes < 60) {
      return (
        minutes +
        "m " +
        remainder +
        "s"
      );
    }

    return (
      Math.floor(minutes / 60) +
      "h " +
      (minutes % 60) +
      "m"
    );
  }

  // Start read-only status polling after the controller is initialized.
  setInterval(
    pollStatus,
    POLL_MS
  );

  pollStatus();

  // Hide the boot overlay after the page has loaded.
  window.addEventListener(
    "load",
    () => {

      setTimeout(
        () => {

          boot.classList.add("hide");

          haptic("success");

        },
        1100
      );
    }
  );

})();
</script>

</body>
</html>
)HTML_PAGE";

// 16. Network initialization

/**
 * The fallback Wi-Fi access point preserves local rover control when the configured
 * station network cannot be reached. This keeps the controller accessible without
 * requiring another router.
 */
void startAccessPoint() {

  // AP mode is sufficient because the ESP32 will host the controller locally.
  WiFi.mode(WIFI_AP);

  // Fixed local address keeps the fallback controller address predictable.
  IPAddress localIP(192, 168, 9, 1);

  // The gateway points back to the ESP32 because it owns the AP network.
  IPAddress gateway(192, 168, 9, 1);

  // Standard /24 subnet for the local rover network.
  IPAddress subnet(255, 255, 255, 0);

  // Apply the fixed AP network configuration before starting the AP.
  const bool configResult =
    WiFi.softAPConfig(
      localIP,
      gateway,
      subnet
    );

  // Start the fallback AP with the configured SSID, password, channel, and
  // client limit.
  const bool apStarted =
    WiFi.softAP(
      Config::AP_SSID,
      Config::AP_PASSWORD,
      Config::WIFI_CHANNEL,
      false,
      Config::MAX_CLIENTS
    );

  // Read the final AP address reported by the Wi-Fi stack.
  const IPAddress ip =
    WiFi.softAPIP();

  Serial.println();
  Serial.println(F("=========================================="));
  Serial.println(F(" AGRIROVER ACCESS POINT MODE"));
  Serial.println(F("=========================================="));
  Serial.print(F(" SSID:       "));
  Serial.println(Config::AP_SSID);
  Serial.print(F(" Password:   "));
  Serial.println(Config::AP_PASSWORD);
  Serial.print(F(" IP:         "));
  Serial.println(ip);
  Serial.print(F(" Web UI:     http://"));
  Serial.println(ip);
  Serial.println(F(" mDNS:       http://agrirover.local"));
  Serial.print(F(" AP config:  "));
  Serial.println(configResult ? F("SUCCESS") : F("FAILED"));
  Serial.print(F(" AP start:   "));
  Serial.println(apStarted ? F("SUCCESS") : F("FAILED"));
  Serial.println(F("=========================================="));
}

/**
 * The firmware first attempts the configured station network. If the connection does not
 * complete within the configured attempt count, the firmware starts AP mode.
 */
void startWiFi() {

  Serial.print(F("Connecting to WiFi: "));
  Serial.println(Config::STA_SSID);

  // Start in station mode so the ESP32 first attempts to join the configured LAN.
  WiFi.mode(WIFI_STA);

  // Set the device hostname before starting the station connection.
  WiFi.setHostname(Config::MDNS_HOSTNAME);

  // Begin the station connection.
  WiFi.begin(
    Config::STA_SSID,
    Config::STA_PASSWORD
  );

  // Track bounded connection attempts so startup cannot wait forever.
  uint8_t attempts = 0;

  while (
    WiFi.status() != WL_CONNECTED &&
    attempts < Config::WIFI_CONNECT_ATTEMPTS
  ) {

    delay(Config::WIFI_RETRY_DELAY_MS);

    Serial.print(".");
    attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println();
    Serial.println(F("=========================================="));
    Serial.println(F(" AGRIROVER WIFI CONNECTED"));
    Serial.println(F("=========================================="));
    Serial.print(F(" SSID:       "));
    Serial.println(Config::STA_SSID);
    Serial.print(F(" IP:         "));
    Serial.println(WiFi.localIP());
    Serial.print(F(" RSSI:       "));
    Serial.print(WiFi.RSSI());
    Serial.println(F(" dBm"));
    Serial.println(F(" Web UI:     http://<ESP32-IP>"));
    Serial.println(F(" mDNS:       http://agrirover.local"));
    Serial.println(F("=========================================="));

  } else {

    // The station network was not available within the startup budget.
    Serial.println();
    Serial.println(F("WiFi connection failed."));
    Serial.println(F("Starting AgriRover Access Point..."));

    startAccessPoint();
  }
}

// 17. Hardware initialization

/**
 * The firmware configures every rover GPIO pin and places the chassis in a known stopped state.
 */
void initializeHardware() {

  // Configure the left H-bridge inputs as outputs.
  pinMode(Pins::LEFT_IN1, OUTPUT);
  pinMode(Pins::LEFT_IN2, OUTPUT);

  // Configure the right H-bridge inputs as outputs.
  pinMode(Pins::RIGHT_IN1, OUTPUT);
  pinMode(Pins::RIGHT_IN2, OUTPUT);

  // Configure the headlight pin as an output.
  pinMode(Pins::STATUS_LED, OUTPUT);

  // Stop the drive system before any network service becomes available.
  stopRover();

  // Start with the headlight off.
  setLed(false);
}

// 18. mDNS initialization

/**
 * The firmware starts mDNS and advertises the HTTP service through agrirover.local.
 */
void startMDNS() {

  // ESP32 Arduino Core 3.3.x starts mDNS through MDNS.begin(). No MDNS.update()
  // call is required in loop() for this implementation.
  if (MDNS.begin(Config::MDNS_HOSTNAME)) {

    // Advertise the local web server to mDNS-aware clients.
    MDNS.addService(
      "http",
      "tcp",
      Config::HTTP_PORT
    );

    Serial.println(
      F("mDNS started: http://agrirover.local")
    );

  } else {

    Serial.println(
      F("mDNS failed to start.")
    );
  }
}

// 19. Firmware setup

/**
 * The firmware initializes serial logging, hardware, networking, mDNS, HTTP routes, and the movement
 * watchdog are initialized in a deterministic order.
 */
void setup() {

  // Start the serial console for startup and operational status messages.
  Serial.begin(115200);

  // Give the USB serial interface a short time to become available.
  delay(200);

  Serial.println();
  Serial.println(F("=========================================="));
  Serial.println(F("        AGRIROVER ESP32 STARTUP"));
  Serial.println(F("=========================================="));
  Serial.println(F(" Team:        Innovex"));
  Serial.println(F(" SIH Challenge: SH26180"));
  Serial.println(F(" Organization: Qualcomm Inc"));
  Serial.println(F(" Category:    Hardware"));
  Serial.println(F(" Theme:       Disaster Management"));
  Serial.println(F("=========================================="));

  // Configure every physical output and put the rover into a known safe state.
  initializeHardware();

  // Bring up station mode or the fallback AP.
  startWiFi();

  // Start local name resolution after the final network interface is ready.
  startMDNS();

  // Register the web controller and all device endpoints.
  registerRoutes();

  // Start accepting HTTP requests.
  server.begin();

  Serial.println(F("HTTP server started."));

  // Arm the movement watchdog from a stopped state.
  touchWatchdog();

  Serial.println(F("Motor safety watchdog armed."));
  Serial.println();

  Serial.println(F("=========================================="));
  Serial.println(F(" AGRIROVER READY"));
  Serial.println(F("=========================================="));
  Serial.println(F(" Team: Innovex"));
  Serial.println(F(" Controller: Web UI"));
  Serial.println(F(" Steering polarity: configured in MotorConfig"));
  Serial.println(F("=========================================="));
}

// 20. Main control loop

/**
 * The main loop services HTTP requests and the local movement watchdog.
 * The loop remains deliberately small, while motor safety stays independent of
 * the browser because serviceWatchdog() executes locally on the ESP32.
 */
void loop() {

  // Process pending HTTP control and status requests.
  server.handleClient();

  // Enforce the local movement timeout independently of network state.
  serviceWatchdog();

  // Do not add MDNS.update() here. ESP32 Arduino Core 3.3.x does not expose
  // that API in the form used by older examples.
}
