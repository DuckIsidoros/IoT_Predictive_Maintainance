#include <Arduino.h>

void setup()
{
    Serial.begin(115200);
    Serial.println("Program stopped");

    while (true)
    {
        delay(1000);
    }
}

void loop()
{
}