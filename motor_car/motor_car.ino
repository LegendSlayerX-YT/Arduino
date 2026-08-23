// AFMotor R4 Compatible Library - Basic Motor Test
// Hardware Setup: DC Motor connected to M1


#include "speed.h"

#if defined(ARDUINO_UNOWIFIR4)
#include "WiFiS3_Wrapper.h"
using MyWifi = MyWifiS3;
#else
#include "WiFiNINA_Wrapper.h"
#endif

MyWifi wifi;

int motor_idx[4]={3, 4, 2, 1};
int low_bound[4]={
    30, 30, 
    30, 30};
int high_bound[4]={
    220, 220,
    220, 220};
SpeedControl motor_control(
  motor_idx,
  low_bound,
  high_bound
);

Speed speed(0,0,0,0);

const unsigned long MOVE_TIMEOUT_MS = 300;
unsigned long move_deadline = 0;
bool moving = false;

void start_move(const Speed& new_speed) {
  speed = new_speed;
  move_deadline = millis() + MOVE_TIMEOUT_MS;
  moving = true;
}

void stop_move() {
  speed = Speed(0,0,0,0);
  moving = false;
}

float parse_query_param(const String& data, const String& key) {
  String search = key + "=";
  int valueStart = data.indexOf(search);
  while (valueStart > 0 && data[valueStart - 1] != '?' && data[valueStart - 1] != '&') {
    valueStart = data.indexOf(search, valueStart + 1);
  }
  if (valueStart < 0) {
    return 0;
  }
  valueStart += search.length();
  int valueEnd = data.indexOf('&', valueStart);
  if (valueEnd < 0) {
    valueEnd = data.length();
  }
  return data.substring(valueStart, valueEnd).toFloat();
}

float clamp(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

void setup() {
  Serial.begin(9600);
  Serial.println("Basic Motor Test!");

#if defined(ARDUINO_UNOWIFIR4)
  motor_control.begin();
#endif

  wifi.connect("MOTO1AAE", "8xpf8x4uru");

  wifi.start_server();
}

void loop() {

  if (moving && millis() >= move_deadline) {
    stop_move();
    motor_control.apply_speed(speed);
  }

  Request request = wifi.get_request();
  if (request.code != -1) {
    Serial.println(request.code);
    Serial.println(request.method);
    Serial.println(request.data);


    if (request.code == 200 && request.data == "/up") {
      start_move(speed + FORWARD_MOVE * 0.3);
    }
    else if (request.code == 200 && request.data == "/down") {
      start_move(speed - FORWARD_MOVE * 0.3);
    }
    else if (request.code == 200 && request.data == "/left") {
      start_move(speed + ROTATE_LEFT * 0.15);
    }
    else if (request.code == 200 && request.data == "/right") {
      start_move(speed - ROTATE_LEFT * 0.15);
    }
    else if (request.code == 200 && request.data == "/shift_left") {
      start_move(speed + SHIFT_LEFT * 0.3);
    }
    else if (request.code == 200 && request.data == "/shift_right") {
      start_move(speed - SHIFT_LEFT * 0.3);
    }
    else if (request.code == 200 && request.data == "/stop") {
      stop_move();
    }
    else if (request.code == 200 && request.data.startsWith("/move")) {
      float forward = clamp(parse_query_param(request.data, "forward"), -1, 1);
      float left = clamp(parse_query_param(request.data, "left"), -1, 1);
      float shift_left = clamp(parse_query_param(request.data, "shift_left"), -1, 1);
      if (forward == 0 && left == 0 && shift_left == 0) {
        stop_move();
      } else {
        start_move(FORWARD_MOVE * forward + ROTATE_LEFT * left + SHIFT_LEFT * shift_left);
      }
    }

    wifi.send_response(String("Hello Chen") +request.data + String(" -- ")+ speed[0] + String(", ") + speed[1] + String(", ")+ speed[2] + String(", ")+ speed[3] + String(", "));
    motor_control.apply_speed(speed);
  }

  delay(20);
}
