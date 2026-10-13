#include <SoftwareSerial.h>
#include <DFRobotDFPlayerMini.h>

SoftwareSerial dfSerial(11, 10); // Arduino RX, TX
DFRobotDFPlayerMini myDFPlayer;

void setup() {
  Serial.begin(115200);   // Serial Monitor

  // Work around the Nano Every SoftwareSerial TX initialization bug by
  // setting the TX line to the UART idle state before starting the port.
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);
  dfSerial.begin(9600);   // DFPlayer

  delay(2000);

  if (!myDFPlayer.begin(dfSerial)) {
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
    return;
  }

  Serial.println("DFPlayer initialized");
  myDFPlayer.volume(15);
  myDFPlayer.playFolder(1, 1); 
  Serial.println(F("Playing 01/001"));
}

void loop() {
}
