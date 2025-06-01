# 🔍Filter Citra

**Tugas Mata Kuliah Pengolahan Citra. dengan menggunakan python dan Boostrap 5.3**

![image alt](https://github.com/AhmdMaulidan/Filter-Citra/blob/433a319ef0202481f67581fcded94ff3a3ea88f8/contoh%20image.png)

# Program Filter Citra

Aplikasi web ini merupakan **program pengolahan citra digital** yang dilengkapi berbagai fitur pemrosesan dan filter gambar. Web ini dirancang untuk memudahkan pengguna dalam menerapkan algoritma deteksi tepi dan morfologi pada gambar.

## 🔧 Fitur Utama

Berikut daftar fitur yang tersedia di sidebar aplikasi :

### 1. **Deteksi Tepi**
- **Threshold**: Mengubah gambar ke dalam bentuk biner (hitam-putih) berdasarkan ambang batas tertentu.
- **Sobel**: Deteksi tepi berdasarkan perubahan intensitas secara horizontal dan vertikal.
- **Prewitt**: Filter deteksi tepi mirip Sobel, dengan pendekatan sederhana terhadap gradien.
- **Roberts**: Deteksi tepi berdasarkan diferensiasi diagonal gambar.
- **Canny**: Deteksi tepi dengan algoritma multi-tahapan yang lebih akurat.

### 2. **Operasi Morfologi**
- **Erosi**: Mengurangi objek putih dalam gambar biner.
- **Dilusi (Dilasi)**: Menebalkan atau memperbesar objek putih dalam gambar biner.

### 3. **Morfologi Lanjut**
- **Morph Boundary**: Menampilkan batas objek dengan operasi morfologi.
- **Morph Regionfilling**: Mengisi bagian dalam objek tertutup.
- **Morph Skeletonizing**: Menipiskan objek hingga bentuk kerangkanya.
- **Morph ConvexHull**: Menghitung dan menampilkan bentuk cembung (convex hull) dari objek.
- **Morph Prunning**: Menghilangkan cabang kecil pada skeleton objek.
- **Morph Thickening**: Menebalkan objek dalam bentuk morfologi.
- **Morph Thinning**: Menipiskan objek hingga hanya garis tengah yang tersisa.

## 🚀 Teknologi yang Digunakan
- Python (Flask)
- OpenCV
- HTML/CSS
- JavaScript(jika digunakan)

##### build with love❤️

