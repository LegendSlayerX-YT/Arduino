#ifndef LIBRARY_WIFININA_WRAPPER_H
#define LIBRARY_WIFININA_WRAPPER_H

#include <SPI.h>
#include <WiFiNINA.h>

struct Request { 
    // HTTP code, -1 means no data
    int code;
    String method;
    String data; 
};

class MyWifi {
  public:
    
    WiFiServer server;

    // The client sending a request to our server.
    WiFiClient incoming_client; 

    void printWifiStatus() {
        // print the SSID of the network you're attached to:
        Serial.print("SSID: ");
        Serial.println(WiFi.SSID());

        // print your board's IP address:
        IPAddress ip = WiFi.localIP();
        Serial.print("IP Address: ");
        Serial.println(ip);

        // print the received signal strength:
        long rssi = WiFi.RSSI();
        Serial.print("signal strength (RSSI):");
        Serial.print(rssi);
        Serial.println(" dBm");
    }

    Request parse_method_data(const String& firstLine) {
        int firstSpace = firstLine.indexOf(' ');
        int secondSpace = firstLine.indexOf(' ', firstSpace + 1);

        if (firstSpace < 0 || secondSpace < 0) {
            return Request {
                .code=400,
                .method="",
                .data=""
            };
        }

        return Request {
            .code=200,
            .method=firstLine.substring(0, firstSpace),
            .data=firstLine.substring(firstSpace + 1, secondSpace)
        };
    }

  public:
    MyWifi() {
    }

    void connect(const char* ssid, const char* pass) {
        // check for the WiFi module:
        if (WiFi.status() == WL_NO_MODULE) {
            Serial.println("Communication with WiFi module failed!");
            // don't continue
            while (true);
        }

        String fv = WiFi.firmwareVersion();
        if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
            Serial.println("Please upgrade the firmware");
        }

        // attempt to connect to WiFi network:
        int status = WL_IDLE_STATUS;
        while (status != WL_CONNECTED) {
            Serial.print("Attempting to connect to SSID: ");
            Serial.println(ssid);
            // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
            status = WiFi.begin(ssid, pass);

            Serial.println("Wait for 10 Seconds.");
            // wait 10 seconds for connection:
            delay(10000);
            Serial.println("Waked up.");
        }
        Serial.println("Wifi Connected.");
        printWifiStatus();
    }

    void start_server() {
        server = WiFiServer(80);
        server.begin();
    }

    // Get the current request received by the server;
    Request get_request() {
        incoming_client = server.available();
        if (!incoming_client.operator bool()) {
            Serial.println("No client");
            return Request{
                .code=-1, 
                .method="",  
                .data=""};
        }

        Serial.println("new request received");

        String firstLine = "";
        String currentLine = "";
        while (incoming_client.connected()) {
            if (incoming_client.available()) {
                char c = incoming_client.read();
                //Serial.write(c);
                // if you've gotten to the end of the line (received a newline
                // character) and the line is blank, the HTTP request has ended,
                // so you can send a reply
                if (c == '\n' && currentLine.length()==0) {
                    return parse_method_data(firstLine);
                }
                if (c == '\n') {
                    // you're starting a new line
                    if (firstLine.length()==0) {
                        firstLine = currentLine;
                    }
                    currentLine = "";
                } else if (c != '\r') {
                    // you've gotten a character on the current line
                    currentLine += c;
                }
            }
        }

        return Request{
            .code=400, 
            .method="",  
            .data=""
        };
    }
    
    void send_response(const String& response) {
        if (incoming_client) {
            incoming_client.println("HTTP/1.1 200 OK");
            incoming_client.println("Content-Type: text/html");
            incoming_client.println("Connection: close");  // the connection will be closed after completion of the response
            incoming_client.println();
            incoming_client.println(response);
            incoming_client.println();
            incoming_client.println();
            
            incoming_client.stop();
            Serial.println("client disconnected");
        }
    }


};

#endif
