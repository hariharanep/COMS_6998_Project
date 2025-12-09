import os
  # accessible as a variable in index.html:
from flask import Flask, request, render_template
from flask_cors import CORS
from workflow.gpt_5 import invoke_gpt_5

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)
CORS(app)

@app.route('/')
def index():
  return render_template("index.html")

@app.post("/llm")
def llm():
    body = request.get_json()
    model = body["model"]
    prompt = body["prompt"]

    if model == "gpt-5":
      return {
        "response": invoke_gpt_5(prompt)
      }
    
    return {
      "response": ""
    }


if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8112, type=int)
  def run(debug, threaded, host, port):
    HOST, PORT = host, port
    print("running on %s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)

  run()