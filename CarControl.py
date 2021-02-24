import serial
from time import sleep
import time
import numpy as np

def angle_control(angle):
    ser = serial.Serial(
    port = '/dev/ttyAMA0',
    baudrate = 9600,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 5
)
    if (angle < 45):
        angle = 45
    elif (angle > 125):
        angle = 125
    send = 'S-' + str(angle) + '\r'
    ser.write(send.encode())
    ser.flush()
    ser.close()
    return angle

def speed_control(speed):
    ser = serial.Serial(
    port = '/dev/ttyAMA0',
    baudrate = 9600,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
    bytesize = serial.EIGHTBITS,
    timeout = 5
)
    if speed < 0:
        speed = 0
    if  speed > 4:
        speed = 4
    send = 'M-F-' + str(speed) + '\r'
    ser.write(send.encode())
    ser.flush()
    ser.close()
    return speed

#su dung tich luy 5 frame
error_arr = np.zeros(5)
t = time.time()
#def PID(error, p= 0.2, i =0.01, d = 0.02):
def PID(error, p= 0.3, i =0.01, d = 0.05):
    global error_arr, t
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error*p
    delta_t = time.time() - t
    t = time.time()
    D = (error-error_arr[1])/delta_t*d
    I = np.sum(error_arr)*delta_t*i
    angle = P + I + D
    #print(P, I, D)

    if abs(angle)>35:
        angle = np.sign(angle)*35
    return -int(angle)

print('hello!')