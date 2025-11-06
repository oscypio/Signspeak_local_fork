# SignSpeak

**Backend**
- Just run **docker compose up** and the docker container should be running at the 8080 port of your machine. (Of course, install docker)

- Format of the data being sent by Jinet:
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


- Format of the data I'm sending to Michal & Mohammed. I added two fields: 
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

**Might be cool to have the same fields name in the ML system. (So, If a field is not needed, you can just ignore it and I will not need to modify the fields name again 😈)**

- I assume the ML client url to send the data to is: http://localhost:8081/ml/process


According to that: 

**The ML Client service that will be defined in the docker compose file should expose 8081, and the endpoints in the code should be "/ml/process"**