/*
 * Teensy Serial-Controlled Pulse Generator
 * 
 * Serial Commands:
 * - START <frequency>  : Start pulse at specified frequency (e.g., "START 100")
 * - STOP               : Stop pulse generation
 * 
 * Response:
 * - OK START <frequency>
 * - OK STOP
 * - ERROR <message>
 */

// --- Configuration ---
const int OUTPUT_PIN = 3;  // Output Pulse

// --- Variables ---
IntervalTimer myTimer;
volatile int outputState = LOW;
volatile bool isRunning = false;
float currentFrequency = 0.0;

// --- Timer ISR (Handles the Pulse Wave) ---
void timerCallback() {
  outputState = !outputState;
  digitalWriteFast(OUTPUT_PIN, outputState);
}

// --- Command Processing ---
void processCommand(String command) {
  command.trim();
  command.toUpperCase();
  
  if (command.startsWith("START")) {
    // Parse frequency from command
    int spaceIndex = command.indexOf(' ');
    if (spaceIndex == -1) {
      Serial.println("ERROR Missing frequency parameter");
      return;
    }
    
    String freqStr = command.substring(spaceIndex + 1);
    float frequency = freqStr.toFloat();
    
    if (frequency <= 0.0 || frequency > 10000.0) {
      Serial.println("ERROR Invalid frequency (must be 0-10000 Hz)");
      return;
    }
    
    // Stop existing pulse if running
    if (isRunning) {
      myTimer.end();
    }
    
    currentFrequency = frequency;
    isRunning = true;
    
    // Set line High immediately to align phase
    outputState = HIGH;
    digitalWriteFast(OUTPUT_PIN, HIGH);
    
    // Start timer for the next toggle
    // frequency Hz -> period = 1/freq seconds -> toggle every period/2
    float microsecInterval = (1000000.0 / frequency) / 2.0;
    myTimer.begin(timerCallback, microsecInterval);
    
    Serial.print("OK START ");
    Serial.println(frequency);
    
  } else if (command.equals("STOP")) {
    if (isRunning) {
      myTimer.end();
      isRunning = false;
      currentFrequency = 0.0;
      digitalWriteFast(OUTPUT_PIN, LOW); // Force Low
      Serial.println("OK STOP");
    } else {
      Serial.println("OK STOP (already stopped)");
    }
    
  } else {
    Serial.print("ERROR Unknown command: ");
    Serial.println(command);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, LOW);
  
  // Send ready message
  Serial.println("READY Teensy Pulse Generator");
}

void loop() {
  // Process serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }
}