# SignSpeak

## I. **Frontend**
The frontend is a **React (Vite)** application that manages the entire user experience and client-side processing. It runs entirely in the user's browser.

### Key Responsibilities

* **Webcam Capture:** Accesses the user's camera via `navigator.mediaDevices.getUserMedia`.
* **Landmark Extraction:** Uses the **MediaPipe** library (running client-side) to extract hand landmarks from the video stream in real-time. **No raw video ever leaves the client**.
* **State Management:** Manages the application's state machine (`IDLE`, `CONNECTING`, `LISTENING`, `PROCESSING`, `ERROR`) to provide clear user feedback and disable buttons when necessary.
* **Real-time Communication:** Uses **SockJS** and **STOMP** to stream the compact landmark data to the Backend (on the `/app/frame` destination) and subscribe to the final translations (from the `/topic/status` topic).
* **UI Rendering:** Displays the live video feed, status indicators, and the final translated text.

### Running in Development Mode (Without Docker)
1.  **Ensure the Backend is Running:** The backend (started via `docker compose up`) must be active on `localhost:8080` for the WebSocket to connect.
2.  Navigate to the frontend directory:
    ```bash
    cd singspeak-frontend
    ```
3.  Install the dependencies:
    ```bash
    npm install
    ```
4.  Start the development server:
    ```bash
    npm run dev
    ```
5.  The application will be available at `http://localhost:5173` (or the port shown by Vite).

## II. **Backend**
- Just run **docker compose up** and the docker container should be running at the 8080 port of your machine. (Of course, install docker)

- Format of the data being sent by the Frontend:
<pre>
{
  "timestamp": 31034.30000001192,
  "landmarks": [
    [
      {
        "x": 0.2482806146144867,
        "y": 0.9458644986152649,
        "z": -1.7798754470277345e-8,
        "visibility": 0
      },
      {
        "x": 0.808993935585022,
        "y": 0.7315307259559631,
        "z": -0.04298790544271469,
        "visibility": 0
      }
    ]
  ],
  "handedness": [
    [
      {
        "score": 0.98486328125,
        "index": 0,
        "categoryName": "Right",
        "displayName": "Right"
      }
    ],
    [
      {
        "score": 0.985992431640625,
        "index": 1,
        "categoryName": "Left",
        "displayName": "Left"
      }
    ]
  ]
}
</pre>


- Format of the data I'm sending to ML part. I added two fields: 
    - sequenceNumber: rank in the list sent. 
    - receivedAt-the moment it's received in the backend:
<pre>
{
  "timestamp": 31034.30000001192,
  "sequenceNumber": 1,
  "receivedAt": 31034.30000001292
  "landmarks": [
    [
      {
        "x": 0.2482806146144867,
        "y": 0.9458644986152649,
        "z": -1.7798754470277345e-8,
        "visibility": 0
      },
      {
        "x": 0.808993935585022,
        "y": 0.7315307259559631,
        "z": -0.04298790544271469,
        "visibility": 0
      }
    ]
  ],
  "handedness": [
    [
      {
        "score": 0.98486328125,
        "index": 0,
        "categoryName": "Right",
        "displayName": "Right"
      }
    ],
    [
      {
        "score": 0.985992431640625,
        "index": 1,
        "categoryName": "Left",
        "displayName": "Left"
      }
    ]
  ]
}
</pre>

1.  Configurations > Frontend:
- Webscoket endpoint: '/ws'
- Make sure the stompClient subscribe to: '/topic/status'
- Publish the data to this endpoint: '/app/frame'  

<pre>
  connect() {
    const socket = new SockJS('http://localhost:8080/ws');
    
    this.stompClient = new Client({
      webSocketFactory: () => socket as any,
      debug: (str) => {
        console.log('STOMP: ' + str);
      },
      reconnectDelay: 5000,
      heartbeatIncoming: 4000,
      heartbeatOutgoing: 4000,
    });
    
    this.stompClient.onConnect = () => {
      console.log('Connected to WebSocket');
      this.isConnected = true;
      this.addStatus('Connected to server');
      
      this.stompClient?.subscribe('/topic/status', (message: IMessage) => {
        this.addStatus('Server: ' + message.body);
      });
    };
    
    this.stompClient.onStompError = (frame) => {
      console.error('Broker error: ' + frame.headers['message']);
      console.error('Details: ' + frame.body);
      this.isConnected = false;
      this.addStatus('Connection error');
    };
    
    this.stompClient.activate();
  }
</pre>

And when sending:

<pre>
this.stompClient.publish({
  destination: '/app/frame',
  body: JSON.stringify(frameData)
});
</pre>

2.  Configurations > ML System:

**Might be cool to have the same fields name in the ML system. (So, If a field is not needed, you can just ignore it and I will not need to modify the fields name again 😈)**

- I assume the ML client url to send the data to is: http://localhost:8081/ml/process


According to that: 

**The ML Client service that will be defined in the docker compose file should expose 8081, and the endpoints in the code should be "/ml/process"**


## III. **ML System**
<!-- Add your comment here -->