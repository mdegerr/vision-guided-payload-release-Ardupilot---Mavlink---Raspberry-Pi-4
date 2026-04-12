import time


class ServoController:
    def __init__(
        self,
        gpio,
        red_pin,
        blue_pin,
        pwm_freq,
        period_us,
        red_init_us,
        red_release_us,
        blue_init_us,
        blue_release_us,
    ):
        self.gpio = gpio
        self.red_pin = red_pin
        self.blue_pin = blue_pin
        self.pwm_freq = pwm_freq
        self.period_us = period_us
        self.red_init_us = red_init_us
        self.red_release_us = red_release_us
        self.blue_init_us = blue_init_us
        self.blue_release_us = blue_release_us
        self._active_pwm = {}

    def setup(self):
        self.gpio.setwarnings(False)
        self.gpio.setmode(self.gpio.BCM)
        self.gpio.setup(self.red_pin, self.gpio.OUT)
        self.gpio.setup(self.blue_pin, self.gpio.OUT)

    def _us_to_duty(self, pulse_width_us):
        pulse = max(500, min(2500, int(pulse_width_us)))
        return (pulse / self.period_us) * 100.0

    def move_to_us(self, pin, pulse_us, duration_sec=0.6, keep_on=False):
        pwm = None
        try:
            pwm = self.gpio.PWM(pin, self.pwm_freq)
            pwm.start(self._us_to_duty(pulse_us))
            time.sleep(max(0.05, float(duration_sec)))
            if keep_on:
                self._active_pwm[pin] = pwm
            else:
                pwm.stop()
        except Exception as exc:
            print(f"Servo control error on pin {pin}: {exc}")
            if pwm is not None:
                try:
                    pwm.stop()
                except Exception:
                    pass

    def initialize(self):
        self.move_to_us(self.red_pin, self.red_init_us)
        self.move_to_us(self.blue_pin, self.blue_init_us)

    def release(self, pin):
        if pin == self.red_pin:
            self.move_to_us(pin, self.red_release_us)
        elif pin == self.blue_pin:
            self.move_to_us(pin, self.blue_release_us)

    def cleanup(self):
        for pwm in self._active_pwm.values():
            try:
                pwm.stop()
            except Exception:
                pass
        self._active_pwm.clear()
        self.gpio.cleanup()
