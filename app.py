import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_dominant_color(image, k=3):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(img)
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    dominant = colors[np.argmax(counts)]
    return dominant

def classify_waste(image_path):
    pil_img = Image.open(image_path).convert('RGB').resize((150, 150))
    np_img = np.array(pil_img)
    img_cv = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # ✅ 1. ความสว่าง
    brightness = np.mean(np_img)

    # ✅ 2. สีเด่น
    dom_color = get_dominant_color(img_cv)
    r, g, b = dom_color

    # ✅ 3. HSV (ใช้ hue + saturation วิเคราะห์อาหาร)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:, :, 0])
    mean_sat = np.mean(hsv[:, :, 1])

    # ✅ 4. ตรวจ contour → กล้วยยาวๆ เรียบๆ
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aspect_ratios = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            ratio = max(w/h, h/w)
            aspect_ratios.append(ratio)

    avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1.0

    # ✅ วิเคราะห์รวม
    if (r > 170 and g > 170 and b < 120) or (mean_hue >= 20 and mean_hue <= 40 and mean_sat > 80):
        if avg_aspect_ratio > 1.5:
            return "ขยะอินทรีย์ (ตรวจพบรูปทรงยาวสีเหลืองคล้ายกล้วย)", "bin_green.png"

    if brightness > 180 and mean_sat < 50:
        return "ขยะรีไซเคิล (ขวดแก้ว, พลาสติกใส)", "bin_blue.png"

    if mean_sat < 80 and avg_aspect_ratio < 1.2:
        return "ขยะทั่วไป (ซองขนม, ถุงพลาสติก)", "bin_red.png"

    return "ขยะรีไซเคิล (พลาสติก, โลหะ)", "bin_blue.png"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    bin_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, bin_image = classify_waste(filepath)

    return render_template('index.html', result=result, bin_image=bin_image)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
