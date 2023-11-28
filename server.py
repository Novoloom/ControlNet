from flask import Flask, request, jsonify
from constrainedDiffusor import diffusionProcessorDispatch

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    return jsonify(['Hello World!'])

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Extract Image and Parameters from the request
        controlnet_type = request.files['controlnetType']
        params = request.json['controlnetParameters']  # Assuming parameters are sent as JSON

        # Call the main function from your script
        response = diffusionProcessorDispatch.main(image, params)

        # Return the response
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
