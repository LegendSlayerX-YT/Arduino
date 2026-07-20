// AFMotor R4 Compatible Library - Basic Motor Test
// Hardware Setup: DC Motor connected to M1


#include "speed.h"
#include "WiFiNINA_Wrapper.h"

MyWifi wifi;

void setup() {
  Serial.begin(9600);           
  Serial.println("Basic Motor Test!");

  wifi.connect("MOTO1AAE", "8xpf8x4uru");

  wifi.start_server();
}

void loop() {

  Serial.println("Loop");

  Request request = wifi.get_request();
  if (request.code != -1) {
    Serial.println(request.code);
    Serial.println(request.method);
    Serial.println(request.data);

    wifi.send_response("Hello Chen");
  }

  delay(1000);
  
  /*
  int motor_idx[4]={3, 4, 2, 1};
  int low_bound[4]={50, 50, 
     50, 50};
  int high_bound[4]={204, 220, 
     176, 220};
  SpeedControl motor_control(
    motor_idx,
    low_bound,
    high_bound
  );

  motor_control.apply_speed(FORWARD_MOVE * 0.3);
  
  delay(3000);


  motor_control.apply_speed(ROTATE_LEFT * 0.3);
  
  delay(3000);
  */
}
