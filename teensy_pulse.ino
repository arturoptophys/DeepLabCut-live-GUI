/*
 * Teensy TTL Triggered 100Hz Pulse Gen
 * * Logic:
 * 1. Listens for a RISING edge on Pin 2 (TTL Trigger).
 * 2. Immediately toggles the output state (On/Off).
 * 3. If On: Starts a 100Hz square wave on Pin 3.
 * 4. If Off: Stops the wave and forces Pin 3 LOW.
 */

// --- Configuration ---
const int TRIGGER_PIN = 2; // Input from TTL
const int OUTPUT_PIN = 3;  // Output Pulse
const float FREQUENCY_HZ = 100.0;

// --- Variables ---
IntervalTimer myTimer;
volatile int outputState = LOW;
volatile bool isRunning = false;
volatile unsigned long lastTriggerTime = 0;

// --- Timer ISR (Handles the 100Hz Wave) ---
void timerCallback() {
  outputState = !outputState;
  digitalWriteFast(OUTPUT_PIN, outputState);
}

// --- Trigger ISR (Handles the Input) ---
void onTriggerReceived() {
  // Optional: Simple glitch filter (ignore triggers closer than 50ms)
  // If your TTL source is very clean, you can remove this 'if'.
  if (micros() - lastTriggerTime < 50000) return; 
  lastTriggerTime = micros();

  isRunning = !isRunning;

  if (isRunning) {
    // START 
    // 1. Set line High immediately to align phase with trigger
    outputState = HIGH;
    digitalWriteFast(OUTPUT_PIN, HIGH);
    
    // 2. Start timer for the next toggle (5ms later)
    // 100Hz = 10ms period -> toggle every 5ms
    float microsecInterval = (1000000.0 / FREQUENCY_HZ) / 2.0;
    myTimer.begin(timerCallback, microsecInterval);
    
  } else {
    // STOP
    myTimer.end();
    digitalWriteFast(OUTPUT_PIN, LOW); // Force Low
  }
}

void setup() {
  pinMode(OUTPUT_PIN, OUTPUT);
  
  // Triggers usually drive the line High/Low, so standard INPUT is fine.
  // We trigger on RISING edge (0V -> 5V transition).
  pinMode(TRIGGER_PIN, INPUT); 
  attachInterrupt(digitalPinToInterrupt(TRIGGER_PIN), onTriggerReceived, RISING);
}

void loop() {
  // Main loop is empty; everything is handled by hardware interrupts
  // for maximum precision.
}