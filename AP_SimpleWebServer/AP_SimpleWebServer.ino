/*
  WiFi Web Server LED Blink

  A simple web server that lets you blink an LED via the web.
  This sketch will connect to your existing WiFi network (SECRET_SSID/SECRET_PASS
  in arduino_secrets.h should be your home network's credentials).
  It will then launch a new server and print out the IP address
  to the Serial Monitor. From there, you can open that address in a web browser
  to turn on and off the LED on pin 13.

  If the IP address of your board is yourAddress:
    http://yourAddress/H turns the LED on
    http://yourAddress/L turns it off

  created 25 Nov 2012
  by Tom Igoe

  Find the full UNO R4 WiFi Network documentation here:
  https://docs.arduino.cc/tutorials/uno-r4-wifi/wifi-examples#simple-web-server
 */

#include "WiFiS3_Wrapper.h"

#include "arduino_secrets.h"

///////please enter your sensitive data in the Secret tab/arduino_secrets.h
char ssid[] = SECRET_SSID;        // your network SSID (name)
char pass[] = SECRET_PASS;        // your network password (use for WPA, or use as key for WEP)

int led = LED_BUILTIN;

MyWifiS3 wifi;

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("WiFi Web Server");

  pinMode(led, OUTPUT);      // set the LED pin mode

  wifi.connect(ssid, pass);
  wifi.start_server();
}

void loop() {
  Request request = wifi.get_request();
  if (request.code != 200) {
    return;
  }

  // Check to see if the client request was "GET /H" or "GET /L":
  if (request.data == "/H") {
    digitalWrite(led, HIGH);               // GET /H turns the LED on
  }
  if (request.data == "/L") {
    digitalWrite(led, LOW);                // GET /L turns the LED off
  }

  String response = "<p style=\"font-size:7vw;\">Click <a href=\"/H\">here</a> turn the LED on<br></p>";
  response += "<p style=\"font-size:7vw;\">Click <a href=\"/L\">here</a> turn the LED off<br></p>";
  wifi.send_response(response);
}
