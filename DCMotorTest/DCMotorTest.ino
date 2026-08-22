/*
This is a test sketch for the Adafruit assembled Motor Shield for Arduino v2
It won't work with v1.x motor shields! Only for the v2's with built in PWM
control

For use with the Adafruit Motor Shield v2
---->	http://www.adafruit.com/products/1438

This version cycles through all 4 DC motor ports (M1, M2, M3, M4).
*/

#include <Adafruit_MotorShield.h>

// Create the motor shield object with the default I2C address
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
// Or, create it with a different I2C address (say for stacking)
// Adafruit_MotorShield AFMS = Adafruit_MotorShield(0x61);

// One motor object per port
Adafruit_DCMotor *motors[4] = {
  AFMS.getMotor(1),
  AFMS.getMotor(2),
  AFMS.getMotor(3),
  AFMS.getMotor(4)
};

void testMotor(Adafruit_DCMotor *motor, uint8_t portNum) {
  Serial.print("Testing M");
  Serial.println(portNum);

  motor->run(FORWARD);
  for (uint8_t i = 0; i < 255; i++) {
    motor->setSpeed(i);
    delay(5);
  }
  for (uint8_t i = 255; i != 0; i--) {
    motor->setSpeed(i);
    delay(5);
  }

  motor->run(BACKWARD);
  for (uint8_t i = 0; i < 255; i++) {
    motor->setSpeed(i);
    delay(5);
  }
  for (uint8_t i = 255; i != 0; i--) {
    motor->setSpeed(i);
    delay(5);
  }

  motor->run(RELEASE);
}

void setup() {
  Serial.begin(9600);           // set up Serial library at 9600 bps
  Serial.println("Adafruit Motorshield v2 - DC Motor test (all 4 motors)!");

  if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  Serial.println("Motor Shield found.");

  // Make sure all motors start off, and give them a starting speed
  for (uint8_t m = 0; m < 4; m++) {
    motors[m]->setSpeed(150);
    motors[m]->run(RELEASE);
  }
}

void loop() {
  for (uint8_t m = 0; m < 4; m++) {
    testMotor(motors[m], m + 1);
    delay(500);
  }

  Serial.println("All motors tested. Pausing...");
  delay(1000);
}
