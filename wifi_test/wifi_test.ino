#include <SPI.h>
#include <WiFiNINA.h> // Library specifically for the Uno WiFi Rev2

// Replace with your network credentials
char ssid[] = "MOTO1AAE"; 
char pass[] = "8xpf8x4uru"; 

int status = WL_IDLE_STATUS;
WiFiServer server(80); // Start server on HTTP default port 80

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect (needed for native USB)
  }

  pinMode(LED_BUILTIN, OUTPUT); // Configure built-in LED pin

  // Check for the WiFi module
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true); // Stop execution
  }

  // Attempt to connect to the network
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(ssid);
    status = WiFi.begin(ssid, pass);
    delay(5000); // Wait 5 seconds for connection attempt
  }

  server.begin(); // Start the web server
  
  // Print connection info to Serial Monitor
  Serial.println("Connected to WiFi!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP()); 
}

void loop() {
  WiFiClient client = server.available(); // Listen for incoming clients

  if (client) {                             // If a client connects,
    Serial.println("New Client Connected."); 
    String currentLine = "";                // Hold incoming data from client
    
    while (client.connected()) {            // Loop while client stays connected
      if (client.available()) {             // If there's data to read
        char c = client.read();             // Read a byte
        Serial.write(c);                    // Print it out to the serial monitor
        
        if (c == '\n') {                    // If the byte is a newline character
          // A blank line (two newlines in a row) marks the end of the HTTP request
          if (currentLine.length() == 0) {
            
            // Send standard HTTP response header
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println("Connection: close");
            client.println(); // Mandatory blank line separating header from body
            
            // Web Page HTML Content
            client.println("<!DOCTYPE html><html>");
            client.println("<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
            client.println("<title>Arduino WiFi Server</title></head>");
            client.println("<body><center><h1>Arduino Uno WiFi Rev2 Server</h1>");
            
            // Simple links that append /H or /L to the IP address
            client.println("<p><a href=\"/H\"><button style=\"padding:15px; font-size:20px;\">TURN LED ON</button></a></p>");
            client.println("<p><a href=\"/L\"><button style=\"padding:15px; font-size:20px;\">TURN LED OFF</button></a></p>");
            
            client.println("</center></body></html>");
            client.println(); // End response
            break; 
          } else {    // If you got a newline, clear currentLine
            currentLine = "";
          }
        } else if (c != '\r') {  // If you got anything else but a carriage return,
          currentLine += c;      // add it to the end of the currentLine
        }

        // Process the request if the client clicked a button
        if (currentLine.endsWith("GET /H")) {
          digitalWrite(LED_BUILTIN, HIGH); // Turn LED on
        }
        if (currentLine.endsWith("GET /L")) {
          digitalWrite(LED_BUILTIN, LOW);  // Turn LED off
        }
      }
    }
    
    // Close the connection
    delay(10); // Give browser time to receive the data
    client.stop();
    Serial.println("Client Disconnected.");
  }
}