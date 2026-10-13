// Very much inspired by https://www.dfrobot.com/blog-1462.html by DFRobot Feb 26 2020
// Additions made by Just Baselmans https://www.youtube.com/justbaselmansYT Jan 23 2023

#include "SoftwareSerial.h"
#include "DFRobotDFPlayerMini.h"

// DFPlayer Mini serial connection.
SoftwareSerial mySoftwareSerial(11, 10);  // RX, TX
DFRobotDFPlayerMini myDFPlayer;

// Control pins.
const uint8_t VOLUME_POT_PIN = A0;
const uint8_t PAUSE_BUTTON_PIN = 2;
const uint8_t NEXT_BUTTON_PIN = 3;

// Polling/debounce settings.
const unsigned long DEBOUNCE_MS = 50;
const int VOLUME_HYSTERESIS = 1;

bool isPaused = false;
int lastVolume = -1;
int lastPauseReading = HIGH;
int lastNextReading = HIGH;
int pauseButtonState = HIGH;
int nextButtonState = HIGH;
unsigned long lastPauseDebounceTime = 0;
unsigned long lastNextDebounceTime = 0;

int readVolumeLevel() {
  int rawValue = analogRead(VOLUME_POT_PIN);
  return map(rawValue, 0, 1023, 0, 30);
}

void updateVolume() {
  int newVolume = readVolumeLevel();
  if (lastVolume < 0 || abs(newVolume - lastVolume) >= VOLUME_HYSTERESIS) {
    lastVolume = newVolume;
    myDFPlayer.volume(lastVolume);
    Serial.print(F("Volume: "));
    Serial.println(lastVolume);
  }
}

void handlePauseButton() {
  int reading = digitalRead(PAUSE_BUTTON_PIN);
  
  if (reading == lastPauseReading) {
    Serial.println("pause");
    Serial.println(reading);
    return;
  }

  if (reading == LOW) {
    isPaused = !isPaused;
    if (isPaused) {
      myDFPlayer.pause();
      Serial.println(F("Paused"));
    } else {
      myDFPlayer.start();
      Serial.println(F("Resumed"));
    }
  }

  lastPauseReading = reading;
}

void handleNextButton() {
  int reading = digitalRead(NEXT_BUTTON_PIN);

  if (reading == lastNextReading){
    Serial.println("next");
    Serial.println(reading);
    return;
  }

  if (reading == LOW) {
    myDFPlayer.next();
    isPaused = false;
    Serial.println(F("Next track"));
  }

  lastNextReading = reading;
}

void setup() {
  pinMode(PAUSE_BUTTON_PIN, INPUT_PULLUP);
  pinMode(NEXT_BUTTON_PIN, INPUT_PULLUP);

  Serial.begin(115200);
  Serial.println("Start Waiting");
  delay(10000);
  
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);
  
  mySoftwareSerial.begin(9600);

  Serial.println();
  Serial.println(F("DFRobot DFPlayer Mini"));
  Serial.println(F("Initializing DFPlayer module ... Wait!"));

  if (!myDFPlayer.begin(mySoftwareSerial)) {
    Serial.println(F("Not initialized:"));
    Serial.println(F("1. Check the DFPlayer Mini connections"));
    Serial.println(F("2. Insert an SD card"));

    uint8_t type = myDFPlayer.readType();
    uint16_t parameter = myDFPlayer.read();
    Serial.print(F("Error type: "));
    Serial.print(type);
    Serial.print(F(" ("));
    switch (type) {
      case TimeOut: Serial.print(F("TimeOut - no response from module, check wiring/power")); break;
      case WrongStack: Serial.print(F("WrongStack - corrupted response")); break;
      case DFPlayerCardInserted: Serial.print(F("CardInserted")); break;
      case DFPlayerCardRemoved: Serial.print(F("CardRemoved")); break;
      case DFPlayerCardOnline: Serial.print(F("CardOnline")); break;
      case DFPlayerPlayFinished: Serial.print(F("PlayFinished")); break;
      case DFPlayerError: Serial.print(F("DFPlayerError, see error code")); break;
      case DFPlayerUSBInserted: Serial.print(F("USBInserted")); break;
      case DFPlayerUSBRemoved: Serial.print(F("USBRemoved")); break;
      case DFPlayerUSBOnline: Serial.print(F("USBOnline")); break;
      case DFPlayerCardUSBOnline: Serial.print(F("CardUSBOnline")); break;
      case DFPlayerFeedBack: Serial.print(F("FeedBack")); break;
      default: Serial.print(F("Unknown")); break;
    }
    Serial.println(F(")"));

    if (type == DFPlayerError) {
      Serial.print(F("Error code: "));
      Serial.print(parameter);
      Serial.print(F(" ("));
      switch (parameter) {
        case Busy: Serial.print(F("Busy - no SD card, insert an SD card")); break;
        case Sleeping: Serial.print(F("Sleeping")); break;
        case SerialWrongStack: Serial.print(F("SerialWrongStack")); break;
        case CheckSumNotMatch: Serial.print(F("CheckSumNotMatch")); break;
        case FileIndexOut: Serial.print(F("FileIndexOut")); break;
        case FileMismatch: Serial.print(F("FileMismatch")); break;
        case Advertise: Serial.print(F("Advertise")); break;
        default: Serial.print(F("Unknown")); break;
      }
      Serial.println(F(")"));
    }

    while (true) {
    }
  }

  Serial.println(F("DFPlayer Mini module initialized!"));

  myDFPlayer.setTimeOut(500);
  myDFPlayer.EQ(0);

  updateVolume();
  myDFPlayer.playFolder(1, 1);  // Plays SD:/01/001.mp3
  Serial.println(F("Playing 01/001"));
}

void loop() {
  updateVolume();
  handlePauseButton();
  handleNextButton();
  delay(100);
}
