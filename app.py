from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from skimage import filters
import os
from datetime import datetime
from scipy.spatial import ConvexHull

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filename, operation):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if operation == 'threshold':
        # Thresholding biner dengan ambang 128
        _, processed = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        
    elif operation == 'sobel':
        # Operator Sobel dengan kernel 3x3
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        processed = np.sqrt(sobel_x**2 + sobel_y**2)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    elif operation == 'prewit':
        # Operator Prewitt dengan kernel 3x3
        kernel_x = np.array([[-1, 0, 1], 
                            [-1, 0, 1], 
                            [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], 
                            [ 0,  0,  0], 
                            [ 1,  1,  1]], dtype=np.float32)
        prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
        processed = np.sqrt(prewitt_x**2 + prewitt_y**2)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    elif operation == 'roberts':
        # Operator Roberts dengan kernel 2x2
        kernel_x = np.array([[1, 0], 
                            [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], 
                            [-1, 0]], dtype=np.float32)
        roberts_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
        roberts_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
        processed = np.sqrt(roberts_x**2 + roberts_y**2)
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    elif operation == 'canny':
        # Deteksi tepi Canny dengan threshold 100 dan 200
        processed = cv2.Canny(img, 100, 200)
        
    elif operation == 'erosion':
        _, bw = cv2.threshold(img, int(0.8 * 255), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        processed = cv2.erode(bw, kernel, iterations=1)
        
    elif operation == 'dilation':
        # Dilasi gambar dengan kernel 3x3
        _, bw = cv2.threshold(img, int(0.8 * 255), 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        processed = cv2.dilate(bw, kernel, iterations=1)

    elif operation == 'morph_boundary':
        # Konversi ke biner dengan threshold 127
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Elemen struktural persegi 3x3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Erosi
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # Ekstraksi boundary: binary - eroded
        boundary = cv2.subtract(binary, eroded)
        
        processed = boundary
         
    elif operation == 'morph_regionfilling':
        # Binerisasi
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Invers citra (lubang menjadi putih)
        inv_img = cv2.bitwise_not(binary)

        # Inisialisasi seed image (satu titik di tengah)
        seed = np.zeros_like(binary)
        h, w = binary.shape
        seed[h // 2, w // 2] = 255  # seed di tengah

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        prev = np.zeros_like(seed)
        while True:
            dilated = cv2.dilate(seed, kernel)
            seed = cv2.bitwise_and(dilated, inv_img)
            if np.array_equal(seed, prev):
                break
            prev = seed.copy()

        filled = cv2.bitwise_or(seed, binary)
        processed = filled
    
    
    elif operation == 'morph_skeletonizing':
        # Step 1: Threshold citra ke bentuk biner
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Step 2: Inisialisasi skeleton kosong
        size = np.size(binary)
        skel = np.zeros(binary.shape, np.uint8)

        # Structuring element (cross)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        while not done:
            # Erosi dan dilasi
            eroded = cv2.erode(binary, element)
            temp = cv2.dilate(eroded, element)

            # Ambil perbedaan antara citra asli dan yang dibuka
            temp = cv2.subtract(binary, temp)
            skel = cv2.bitwise_or(skel, temp)

            binary = eroded.copy()

            zeros = size - cv2.countNonZero(binary)
            if zeros == size:
                done = True

        processed = skel
    

    elif operation == 'morph_convexHull':
        # Ubah ke citra biner (threshold)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Temukan kontur
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Buat kanvas kosong
        convex_image = np.zeros_like(binary)

        # Untuk setiap kontur, hitung convex hull dan gambar
        for cnt in contours:
            if len(cnt) >= 3:
                hull = cv2.convexHull(cnt)
                cv2.drawContours(convex_image, [hull], 0, 255, -1)  # isi area convex hull

        processed = convex_image

    elif operation == 'morph_pruning':
        # Step 1: Binarisasi gambar
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Step 2: Skeletonizing dengan thinning dari skimage
        from skimage.morphology import skeletonize, remove_small_objects
        from skimage.util import invert

        # Konversi ke boolean
        binary_bool = binary > 0
        skeleton = skeletonize(binary_bool)

        # Step 3: Hapus spur/cabang kecil (pruning)
        pruned = remove_small_objects(skeleton, min_size=30)  # ukuran ini bisa disesuaikan

        # Konversi ke uint8
        processed = (pruned * 255).astype(np.uint8)

    elif operation == 'morph_thickening':
        # Step 1: Binarisasi citra
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Step 2: Skeletonisasi terlebih dahulu
        from skimage.morphology import skeletonize, dilation, square
        from skimage.util import invert

        # Konversi ke boolean
        binary_bool = binary > 0
        skeleton = skeletonize(binary_bool)

        # Step 3: Thickening — iterasi OR dengan hasil dilasi
        thick = skeleton
        for _ in range(5):  # jumlah iterasi bisa diubah sesuai kebutuhan
            dilated = dilation(thick, square(3))
            thick = np.logical_or(thick, dilated)

        # Konversi ke uint8
        processed = (thick * 255).astype(np.uint8)

    elif operation == 'morph_thinning':
        from skimage.morphology import thin

        # Step 1: Binarisasi
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Step 2: Konversi ke boolean
        binary_bool = binary > 0

        # Step 3: Terapkan thinning
        thinned = thin(binary_bool)

        # Step 4: Konversi ke uint8 untuk disimpan sebagai gambar
        processed = (thinned * 255).astype(np.uint8)

    else:
        processed = img
    
    processed_filename = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, processed)
    
    return processed_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        operation = request.form.get('operation')
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            processed_filename = process_image(filename, operation)
            return {'processed_filename': processed_filename}
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Ganti dengan port lain (5001, 8000, 8080, dll)