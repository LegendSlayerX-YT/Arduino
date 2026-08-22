#ifndef MOTOR_CAR_SPEED_V2_H
#define MOTOR_CAR_SPEED_V2_H

#include <Adafruit_MotorShield.h>

const int LF_IDX=0;
const int RF_IDX=1;
const int LB_IDX=2;
const int RB_IDX=3;

struct Speed {
    float speed[4];

    Speed(float lf_speed, float rf_speed, float lb_speed, float rb_speed) {
        this->speed[0] = lf_speed;
        this->speed[1] = rf_speed;
        this->speed[2] = lb_speed;
        this->speed[3] = rb_speed;
    }

    Speed(const float speed[4]) {
        for (int i = 0; i < 4; i++) {
            this->speed[i] = speed[i];
        }
    }

    float& operator[](int i) {
        return speed[i];
    }

    float operator[](int i) const {
        return speed[i];
    }

    Speed operator*(float coefficient) const {
        float new_speed[4];
        for (int i = 0; i < 4; i++) {
            new_speed[i] = speed[i] * coefficient;
        }

        return Speed(new_speed);
    }

    Speed operator/(float coefficient) const {
        float new_speed[4];
        for (int i = 0; i < 4; i++) {
            new_speed[i] = speed[i] / coefficient;
        }

        return Speed(new_speed);
    }

    Speed operator+(const Speed& speed2) const {
        float new_speed[4];
        for (int i = 0; i < 4; i++) {
            new_speed[i] = speed[i] + speed2.speed[i];
        }

        return Speed(new_speed);
    }

    Speed operator-(const Speed& speed2) const {
        float new_speed[4];
        for (int i = 0; i < 4; i++) {
            new_speed[i] = speed[i] - speed2.speed[i];
        }

        return Speed(new_speed);
    }
};

const Speed FORWARD_MOVE(1,1,1,1);
const Speed ROTATE_LEFT(-1,1,-1,1);
const Speed SHIFT_LEFT(-1,1,1,-1);

struct SpeedControl {
    int low_bound[4];
    int high_bound[4];
    Adafruit_MotorShield shield;
    Adafruit_DCMotor* motor[4];

    SpeedControl(
        int motor_idx[4],
        int low_bound[4],
        int high_bound[4]
    ) {
        for (int i = 0; i < 4; i++) {
            this->low_bound[i] = low_bound[i];
            this->high_bound[i] = high_bound[i];
            motor[i] = shield.getMotor(motor_idx[i]);
        }
    }

    // Must be called from setup(), after Serial/Wire are ready, unlike the
    // AFMotor_R4 version which initializes the controller lazily.
    bool begin() {
        return shield.begin();
    }

    void apply_speed(const Speed& speed) {
        for (int i = 0; i< 4; i++) {
            float abs_speed = min(fabs(speed[i]), 1);
            if (abs_speed < 1e-3) {
                motor[i]->run(RELEASE);
                continue;
            }

            motor[i]->run((speed[i] > 0) ? FORWARD: BACKWARD);
            motor[i]->setSpeed(
                static_cast<int>( (high_bound[i] - low_bound[i]) * abs_speed + low_bound[i] )
            );
        }
    }
};

#endif
