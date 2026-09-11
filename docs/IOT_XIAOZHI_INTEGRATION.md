# SmartCropVision IoT, AgriRover & XiaoZhi Voice Integration

## 1. Physical Architecture Overview

SmartCropVision integrates hardware telemetry and voice interaction through authenticated transport layers while keeping heavy neural network models strictly server side.

```
┌────────────────────────────────────────────────────────┐
│                   IoT Edge Peripherals                 │
│                                                        │
│  [ ESP8266 Weather Node ]    [ AgriRover Robotic Cart ] │
│   * DHT11 Temp & Humidity     * Dual Motor Drive       │
│   * Soil Moisture Sensor      * Micro Camera Stream    │
│   * Rain Detection Board      * Variable Rate Sprayer  │
│   * 16x2 LCD Local Display    * Obstacle Safety Sensor │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ (Authenticated JSON / HTTP)
                            ▼
┌────────────────────────────────────────────────────────┐
│            SmartCropVision Central Backend             │
│                                                        │
│  * Multi Tier AI Inference (Server GPU / CPU)          │
│  * Precision Dosage Multiplier Calculation             │
│  * Model Context Protocol (MCP) Server for Voice       │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ (MCP Tool Calls)
                            ▼
┌────────────────────────────────────────────────────────┐
│               XiaoZhi AI Voice Assistant               │
│                                                        │
│  * Natural Speech Querying for Field Extension Agents  │
│  * Preserves Diagnostic Confidence & Uncertainty       │
│  * Summarizes Agronomic Treatment Advisories           │
└────────────────────────────────────────────────────────┘
```

---

## 2. ESP8266 Environmental Node Firmware

The ESP8266 firmware source is maintained in `iot/esp8266_firmware.ino`.

### 2.1 Hardware Connections
* **Microcontroller**: NodeMCU ESP8266 or Wemos D1 Mini (80 MHz)
* **DHT11 Sensor**: Digital Pin D4 (GPIO2) for air temperature and relative humidity.
* **Analog Soil Moisture Sensor**: Pin A0 (10 bit analog to digital converter) measuring soil resistivity.
* **Rain Drop Sensor**: Digital Pin D5 (GPIO14) signaling binary precipitation.
* **Local Display**: 16x2 Liquid Crystal Display over I2C on Pins D1 (SCL, GPIO5) and D2 (SDA, GPIO4).

### 2.2 Operational Cycle
1. Boots and connects to local greenhouse WiFi.
2. Polls environmental sensors every 5 seconds.
3. Formats telemetry as JSON payload.
4. Transmits telemetry to the central backend at `/api/v1/crops/recommend`.
5. Parses returned advisory text and scrolls recommendation onto the local 16x2 LCD display.

---

## 3. AgriRover Robotic Vehicle Protocol

The AgriRover is an autonomous or semi autonomous four wheel rover designed to inspect canopy rows and perform variable rate chemical intervention.

### 3.1 Separation of High Level AI and Low Level Motor Control
* **The Backend Role**: Analyzes camera frames to locate diseased foliage coordinates, computes lesion density, and calculates the recommended chemical dosage multiplier (for example, 1.25x for severe blight versus 0.0x for healthy foliage).
* **The Firmware Role**: The on rover microcontroller remains strictly responsible for motor pulse width modulation, obstacle avoidance, emergency stopping, and battery safety. High level AI commands never bypass motor safety boundaries.

### 3.2 Acknowledgment Handshake
All actuation commands transmit a unique identifier. A command is only marked as executed when the rover transmits back a confirmation payload containing sensor telemetry and motor encoder position.

---

## 4. XiaoZhi Voice Assistant MCP Protocol

SmartCropVision provides voice capability through the Model Context Protocol (MCP) enabling field extension workers to speak with the diagnostic engine.

### 4.1 Truthful Voice Semantics
* **Preserving Uncertainty in Speech**: If the vision classifier returns a confidence below 50% or flags out of distribution status, the voice assistant states that the diagnosis is uncertain and advises manual inspection.
* **Zero Fabrication**: If no image analysis is active in the current session, the assistant informs the grower that no specimen has been submitted rather than inventing a disease label.
* **Concise Natural Delivery**: Voice responses prioritize immediate agronomic action items (such as pruning infected leaves or adjusting furrow drainage) over complex statistical indices.
