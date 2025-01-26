#include <WiFi.h>
#include <ESP32Servo.h>

Servo myservo; // Create servo object
const int servoPin = 4; // GPIO connected to servo
const char* ssid = "MIT"; // Replace with your WiFi SSID
const char* password = "i%739nKGFT"; // Replace with your WiFi password

WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  myservo.attach(servoPin); // Attach servo with min/max pulse widths
  delay(1000);
  myservo.write(90);
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;

  String request = client.readStringUntil('\r');
  client.flush();

  if (request.indexOf("/servo?angle=") >= 0) {
    int posStart = request.indexOf('=') + 1;
    int posEnd = request.indexOf(' ', posStart);
    String posStr = request.substring(posStart, posEnd);
    int angle = posStr.toInt();
    angle = constrain(angle, 0, 180);
    myservo.write(angle); 
    Serial.print("Servo Position: ");
    Serial.println(angle);
  }

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println();
}