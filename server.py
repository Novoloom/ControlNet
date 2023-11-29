from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from PIL import Image
import numpy as np
import io
import base64
from constrainedDiffusor import diffusionProcessorDispatch


app = Flask(__name__)
CORS(app)

@app.route('/hi', methods=['GET'])
def index():
    return "Hello World!"

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Extracting image file from the request
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        image_array = np.array(image)

        # Extracting JSON data
        controlnet_type = request.form['controlnetType']
        params = json.loads(request.form['controlnetParameters'])

        print("Image received: ", image.filename)
        print("controlnet type: ", controlnet_type)
        print("params: ", params)
        params["input_image"] = image_array

        processed_images = diffusionProcessorDispatch.main(controlnet_type, params)
        base64_images = []
        for img in processed_images:
            pil_img = Image.fromarray(img)
            img_io = io.BytesIO()
            pil_img.save(img_io, format='PNG')
            img_io.seek(0)
            base64_data = base64.b64encode(img_io.read()).decode('utf-8')
            base64_images.append(base64_data)

        return jsonify({"images": base64_images})

    except Exception as e:
        return jsonify({"error": str(e), "trace": str(e.__traceback__)})

if __name__ == '__main__':
    print("Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

