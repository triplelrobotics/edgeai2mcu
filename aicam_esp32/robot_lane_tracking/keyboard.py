from adafruit_httpserver import Request, Response, GET

HTML = """
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Robot Keyboard Control</title>
</head>
<body>
  <h2>Robot Keyboard Control</h2>
  <p>Click this page first, then use keyboard:</p>
  <p>Left Arrow : LEFT</p>
  <p>Up Arrow : STRAIGHT</p>
  <p>Right Arrow : RIGHT</p>
  <p>Down Arrow or Space : STOP</p>
  <p id="state">state: STOP</p>

  <script>
  function sendCmd(cmd){
      fetch("/cmd?m=" + cmd);
      document.getElementById("state").innerText = "state: " + cmd;
  }

  document.addEventListener("keydown", function(e){
      if(e.repeat) return;
      if(e.key === "ArrowLeft"){
          e.preventDefault();
          sendCmd("LEFT");
      }
      if(e.key === "ArrowRight"){
          e.preventDefault();
          sendCmd("RIGHT");
      }
      if(e.key === "ArrowUp"){
          e.preventDefault();
          sendCmd("STRAIGHT");
      }
      if (e.key === "ArrowDown" || e.key === " "){
          e.preventDefault();
          sendCmd("STOP");
      }
  });

  document.addEventListener("keyup", function(e){
      if(e.key === "ArrowLeft" || e.key === "ArrowRight"){
          sendCmd("STRAIGHT");
      }
  });
  </script>
</body>
</html>
"""


def register_routes(server, motor):
    @server.route("/", GET)
    def index(request: Request):
        return Response(request, body=HTML, content_type="text/html")

    @server.route("/cmd", GET)
    def cmd(request: Request):
        action = request.query_params.get("m", "STOP")
        motor.set_action(action)
        return Response(
            request,
            body="OK %s\n" % motor.get_action(),
            content_type="text/plain"
        )

    @server.route("/lane", GET)
    def lane(request: Request):
        action = request.query_params.get("m", "STOP")
        motor.set_lane_action(action)
        body = (
            "OK lane "
            + motor.get_action()
            + " score="
            + request.query_params.get("score", "")
            + " stability="
            + request.query_params.get("stability", "")
            + "\n"
        )
        return Response(request, body=body, content_type="text/plain")

    @server.route("/estop", GET)
    def estop(request: Request):
        motor.estop()
        return Response(request, body="OK ESTOP\n", content_type="text/plain")

    @server.route("/arm", GET)
    def arm(request: Request):
        motor.arm()
        return Response(request, body="OK ARMED STOP\n", content_type="text/plain")

    @server.route("/state", GET)
    def state(request: Request):
        return Response(
            request,
            body=motor.get_action() + "\n",
            content_type="text/plain"
        )

    @server.route("/params", GET)
    def params(request: Request):
        p = motor.get_params()
        body = (
            '{"base":' + str(p["base"]) +
            ',"min_effective":' + str(p["min_effective"]) +
            ',"turn_delta":' + str(p["turn_delta"]) + '}\n'
        )
        return Response(request, body=body, content_type="application/json")

    @server.route("/config", GET)
    def config(request: Request):
        qs = request.query_params

        try:
            base = float(qs["base"]) if "base" in qs else None
            min_effective = float(qs["min_effective"]) if "min_effective" in qs else None
            turn_delta = float(qs["turn_delta"]) if "turn_delta" in qs else None

            motor.set_params(
                base=base,
                min_effective=min_effective,
                turn_delta=turn_delta,
            )

            p = motor.get_params()
            body = (
                "OK "
                "base=" + str(p["base"]) +
                " min_effective=" + str(p["min_effective"]) +
                " turn_delta=" + str(p["turn_delta"]) + "\n"
            )
            return Response(request, body=body, content_type="text/plain")

        except Exception:
            return Response(
                request,
                body="ERR usage: /config?base=0.13&min_effective=0.05&turn_delta=0.07\n",
                content_type="text/plain",
                status=400,
            )
