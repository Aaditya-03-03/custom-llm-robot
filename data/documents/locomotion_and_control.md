# IOFT Robot Locomotion & Control Specification

## Supported Locomotion Commands
The IOFT Humanoid Robot supports five primary physical locomotion and control commands:

1. **FORWARD**: Drives the robot forward in the direction of its current heading.
2. **BACKWARD**: Moves the robot in reverse along its heading.
3. **LEFT**: Pivot turns the robot counterclockwise.
4. **RIGHT**: Pivot turns the robot clockwise.
5. **STOP**: Unconditional emergency stop that cuts motor PWM immediately (permitted regardless of authority mode or connection state).

## Speed Profiles
The robot firmware maps three standardized speed profiles to motor PWM percentages:
- **slow**: 20% PWM / speed level 20. Recommended for precision navigation and approaching obstacles.
- **medium**: 40% PWM / speed level 40. Standard walking speed.
- **fast**: 70% PWM / speed level 70. Maximum operational speed.
- Custom integer speeds between 1 and 100 are also supported when explicitly specified.

## Step Timing & Durations
- Physical steps are calibrated to a baseline duration of **0.5 seconds per step**.
- For example, an instruction for 2 steps corresponds to a continuous drive duration of 1.0 second.
- Default step count is 1 step if unspecified (or resolved via clarification).

## Communication Protocols & Network Ports
- **Motor Control Port (UDP 8888)**: The AI server transmits formatted ASCII control packets:
  `CMD|SPEED|DURATION` (e.g. `FORWARD|20|1.00`) to the ESP32 microcontroller (`rioft.local:8888`).
  The ESP32 responds with acknowledgment string `MOTOR_ACK`.
- **Telemetry & State Port (UDP 8889)**: Receives real-time telemetry packets from the robot containing IMU motion state, ultrasonic range, and battery voltage.
- **Hardware Watchdog**: ESP32 motor firmware enforces a 1000ms safety watchdog timer. If continuous heartbeat/refresh packets cease, the robot automatically cuts motor power.

## Safety & Authority Architecture
1. **Execution Authority Modes**:
   - `MANUAL`: Human operator has sole control. Autonomous AI locomotion commands are rejected with `ASSISTANT_OWNERSHIP_REJECTED`.
   - `AI`: Autonomous AI planning and physical dispatch are authorized via `POST /api/v1/commands/ownership`.
2. **Telemetry Freshness Gate**: Physical movement requires live heartbeat telemetry within the last 5.0 seconds. If disconnected, requests safely fail-closed with `ASSISTANT_ROBOT_UNAVAILABLE`.
3. **Pre-Validation**: All action plans undergo strict schema and parameter bounds checks before any packet transmission.

## API Endpoints for Interaction
- `POST /api/v1/assistant`: Natural language instruction planner and physical locomotion executor.
- `POST /api/v1/chat`: Conversational AI with MongoDB session memory and RAG knowledge grounding.
- `GET /api/v1/robot/state`: Real-time snapshot of connection status, current authority, and observed motion state.
- `POST /api/v1/commands/ownership`: Switch execution authority between `MANUAL` and `AI`.
