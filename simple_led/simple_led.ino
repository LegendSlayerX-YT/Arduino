int LED_PIN = 2;
int LED_PIN2 = 3;
int LED_PIN3 = 5;
int BUTTON_PIN = 4;
int NUM = 0;
int BUTTON_LAST_STATE = LOW;

void setup() {
  // put your setup code here, to run once:
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(LED_PIN, HIGH);
}

int take_nth_binary(int digits[]){
  int number_copy = NUM;
  int counter = 0;
  while(counter < 8){
    digits[counter] = number_copy % 2;
    counter++;
    number_copy /= 2;
  }
}

void loop() {
  // put your main code here, to run repeatedly:
  int curr_state = digitalRead(BUTTON_PIN);
  if(curr_state != BUTTON_LAST_STATE && curr_state == LOW){
    NUM++;
    int digits[8];
    take_nth_binary(digits);
    digitalWrite(LED_PIN, digits[0]);
    digitalWrite(LED_PIN2, digits[1]);
    digitalWrite(LED_PIN3, digits[2]);
  }
  BUTTON_LAST_STATE = curr_state;
  delay(100);
}
