from flask import Flask, request, jsonify

from app.torch_utils import transform_image, get_prediction

app = Flask(__name__)

int2class = {0:'neither', 1:'cat', 2:'dog', }
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# This is an API endpoint, we only have one in this case.
# allowed methods in this case is only POST.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            print('read image!')
            tensor = transform_image(img_bytes)
            print('image transformed!', tensor.size())
            prediction = get_prediction(tensor)
            print('prediction done!')
            data = {'prediction': prediction.item(), 'class_name': int2class[prediction.item()]}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})