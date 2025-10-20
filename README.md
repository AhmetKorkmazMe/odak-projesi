<div align="center">

# Odak Projesi
### Yapay Zeka Destekli Görsel Dikkat Analizi Aracı

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

Odak Projesi, bir görsel veya video üzerindeki insan dikkatini bilimsel metotlarla modelleyen ve analiz eden bir web uygulamasıdır. Tasarımcıların, pazarlama uzmanlarının ve araştırmacıların, oluşturdukları görsel materyallerin hangi bölgelerinin daha dikkat çekici olduğunu veriye dayalı olarak anlamalarını sağlar.

<br>

## 📋 İçindekiler
1. [Analiz Modülleri ve Sonuçların Yorumlanması](#1-analiz-modülleri-ve-sonuçların-yorumlanması)
2. [Sistemin Teknik Mimarisi ve İşleyişi](#2-sistemin-teknik-mimarisi-ve-işleyişi)
3. [Analizlerin Bilimsel Temelleri](#3-analizlerin-bilimsel-temelleri)
4. [Kurulum ve Çalıştırma](#4-kurulum-ve-çalıştırma)
5. [Kaynakça ve Referanslar](#5-kaynakça-ve-referanslar)

---

## 1. Analiz Modülleri ve Sonuçların Yorumlanması

<details>
<summary><strong>► Analizlerin detaylı yorum kılavuzunu görmek için buraya tıklayın</strong></summary>
<br>

Uygulamanın sunduğu analizler ve bu sonuçların tasarımlarınızı iyileştirmek için nasıl yorumlanacağı aşağıda açıklanmıştır.

### a. Isı Haritası (Heatmap)
**Analizin Amacı:** Isı haritası, bir kullanıcının bir görsele ilk baktığında gözünün istemsizce nereye odaklanacağını gösteren bir haritadır. Kırmızı ve sarı "sıcak" bölgeler en çok dikkat çeken alanları, mavi ve yeşil "soğuk" bölgeler ise en az dikkat çeken alanları temsil eder.

**Sonuçların Yorumlanması:**
* **Olumlu Göstergeler:** Sıcak bölgelerin, logonuz, ana başlığınız, ürün görseliniz veya "Satın Al" gibi önemli bir buton üzerinde yoğunlaşması, tasarımınızın mesajını başarıyla ilettiğini gösterir. Web siteleri için, sıcak bölgelerin Nielsen Norman Group tarafından tanımlanan "F-Şeklinde Tarama Deseni" ile uyumlu olması (sayfanın sol üst ve orta kısımlarında yoğunlaşması) genellikle olumlu bir işarettir.
* **İyileştirme Alanları:** Sıcak bölgelerin alakasız bir stok fotoğraf, dekoratif bir element veya önemsiz bir metin üzerinde yoğunlaşması, kullanıcının dikkatinin dağıldığına ve ana mesajınızın gözden kaçtığına işarettir. Kritik elementlerin "soğuk" bölgelerde kalması, tasarım hiyerarşisinin gözden geçirilmesi gerektiğini belirtir.

### b. Bakış Rotası (Gaze Plot)
**Analizin Amacı:** Bu simülasyon, bir kullanıcının gözünün görsel üzerindeki en dikkat çekici noktalar arasında hangi sırayla gezinebileceğini tahmin eder. Rota, en dikkat çekici noktadan (1) başlar ve azalan dikkat sırasına göre devam eder.

**Sonuçların Yorumlanması:**
* **Olumlu Göstergeler:** Rotanın mantıksal bir "görsel hikaye" anlatmasıdır. Örneğin, rota sırasıyla "1. Başlık -> 2. Ürün Görseli -> 3. Buton" gibi bir akışı takip ediyorsa, tasarımınızın yönlendirmesi başarılıdır.
* **İyileştirme Alanları:** Rotanın kaotik olması, önemli elementleri atlaması veya alakasız yerler arasında gidip gelmesi, tasarımınızda görsel bir hiyerarşi ve gruplama sorunu olduğunu gösterir.

### c. CTA Tespiti ve Skorlaması
**Analizin Amacı:** Algoritma, görselinizdeki "Satın Al", "İncele", "Başvur" gibi eyleme çağrı (CTA) butonlarını bulur ve bu butonların genel dikkat çekme potansiyelini 0-100 arasında puanlar.

**Sonuçların Yorumlanması:**
* **Skor > 75 (Çok İyi):** Butonunuz hem metin olarak nettir hem de görsel olarak çok baskındır. Kullanıcıların gözden kaçırma ihtimali çok düşüktür.
* **Skor 50-74 (İyi):** Butonunuz işlevsel ve bulunabilir ancak daha dikkat çekici hale getirilebilir. Rengini, boyutunu veya konumunu gözden geçirebilirsiniz.
* **Skor < 50 (Zayıf):** Butonunuz ya görsel olarak çok sönük kalıyor ya da metni bir eylem içermediği için algoritma tarafından zayıf bir aday olarak görülüyor.

</details>

---

## 2. Sistemin Teknik Mimarisi ve İşleyişi
Bu bölüm, uygulamanın arka planında çalışan teknolojileri ve veri akışını mühendislik bakış açısıyla detaylandırmaktadır.

### Teknoloji Yığını

| Kategori | Teknoloji |
| :--- | :--- |
| **Backend Framework** | Python 3.10, Flask |
| **Görüntü İşleme** | OpenCV-Python 4.x |
| **Optik Karakter Tanıma (OCR)** | Pytesseract |
| **Veri Görselleştirme** | Matplotlib |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Konteynerizasyon**| Docker, Docker Compose |

### Veri Akışı ve Sistem Mimarisi
Uygulama, stabil ve güvenli bir kullanıcı deneyimi için _Post-Redirect-Get (PRG)_ mimari desenini kullanır.

1.  **Dosya Yükleme (POST):** Kullanıcı bir dosya yüklediğinde, `multipart/form-data` olarak ilgili `/upload_...` endpoint'ine gönderilir. Flask, `werkzeug.utils.secure_filename` ile dosya adını sanitize eder ve dosyayı geçici olarak `/static/uploads` dizinine yazar.
2.  **Analiz Süreci Tetikleme:**
    * **Video için:** `process_video` fonksiyonu, videoyu `cv2.VideoCapture` ile okur. Belirli saniye aralıklarıyla (`SAMPLING_INTERVAL_SECONDS`) kareler arasında `cv2.absdiff` ile mutlak fark hesaplar. Bu farkın belirli bir eşik (`CHANGE_THRESHOLD`) değerini geçmesi, o anki karenin "Anahtar Kare" olarak kabul edilmesini sağlar.
    * **Görsel için:** `perform_analysis` fonksiyonu doğrudan çağrılır.
3.  **Çekirdek Analiz (perform_analysis):** Her bir resim (veya anahtar kare), bu merkezi fonksiyon içinde sırasıyla `generate_heatmap`, `generate_focus_map`, `generate_gaze_plot` ve `score_button_candidates` alt fonksiyonlarından geçirilir. Her bir fonksiyonun çıktısı (görsel dosyalar), `/static/outputs` dizinine yazılır.
4.  **Veri Kalıcılığı ve Yönlendirme:** Tüm analiz sonuçları bir Python sözlüğünde toplanır ve Flask'in `session` objesinde saklanır. Ardından, sunucu `redirect(url_for('...'))` ile kullanıcıyı sonuçların gösterileceği yeni bir URL'e yönlendirir.
5.  **Sonuçların Gösterimi (GET):** Kullanıcının tarayıcısı bu yeni URL'e standart bir `GET` isteği yapar. İlgili Flask rotası, `session`'dan sonuç verilerini çeker ve `render_template` ile HTML sayfasını dinamik olarak oluşturarak kullanıcıya sunar.

---

## 3. Analizlerin Bilimsel Temelleri
* **Isı Haritası (Saliency):** Bu modül, insan görsel sisteminin "aşağıdan yukarıya dikkat" (bottom-up attention) mekanizmasını modeller. Temelleri, **Itti, Koch ve Niebur (1998)** tarafından geliştirilen "Saliency-Based Visual Attention" modeline dayanır.
* **Bakış Rotası (Gaze Plot):** İnsan gözünün "sakkad" (hızlı sıçramalar) ve "fiksasyon" (kısa duraklamalar) hareketlerini simüle eder.
* **CTA Tespiti:** Tasarımcı Don Norman'ın **"Olanaklılık" (Affordance)** kavramına dayanır. Bir elementin tasarımı, onun nasıl kullanılacağını ima etmelidir.

---

## 4. Kurulum ve Çalıştırma
Projeyi çalıştırmak için sisteminizde **Git**, **Docker** ve **Docker Compose** kurulu olmalıdır.

1.  **Projeyi Klonlayın:**
    ```bash
    git clone [https://github.com/AhmetKorkmazMe/odak-projesi.git](https://github.com/AhmetKorkmazMe/odak-projesi.git)
    ```
2.  **Proje Dizinine Gidin:**
    ```bash
    cd odak-projesi
    ```
3.  **Uygulamayı Başlatın:**
    ```bash
    docker-compose up -d
    ```
4.  **Erişim:**
    Kurulum tamamlandıktan sonra, web uygulamasına `http://localhost` veya `http://sunucu_ip_adresiniz` adresi üzerinden erişebilirsiniz.

---

## 5. Kaynakça ve Referanslar
* Itti, L., Koch, C., & Niebur, E. (1998). *A model of saliency-based visual attention for rapid scene analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence.
* Hou, X., & Zhang, L. (2007). *Saliency detection: A spectral residual approach.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
* Nielsen, J. (2006). *F-Shaped Pattern For Reading Web Content.* Nielsen Norman Group.
* Norman, D. (2013). *The Design of Everyday Things: Revised and Expanded Edition.* Basic Books.
* Google Tesseract OCR Engine & OpenCV Library.

<br>

<details>
<summary><strong>► Click here for the English Version</strong></summary>
<br>

# Odak Project: An AI-Powered Visual Attention Analysis Platform

The Odak Project is a web application that models and analyzes human visual attention on an image or video using scientific methods. It enables designers, marketing experts, and researchers to understand, based on data, which parts of their visual materials are more engaging.

## 1. Analysis Modules & Interpretation of Results

The analyses provided by the application and how to interpret these results to improve your designs are explained below.

### a. Heatmap
Purpose: A heatmap is a map that shows where a user's eye will involuntarily focus when they first look at an image. "Hot" regions in red and yellow represent the most attention-grabbing areas, while "cold" regions in blue and green represent the least.

Interpretation:
* Positive Indicators: If hot zones are concentrated on your logo, main headline, product image, or a key button like "Buy Now," it indicates that your design successfully conveys its message.
* Areas for Improvement: If hot zones are focused on an irrelevant stock photo or a decorative element, it's a sign that the user's attention is distracted. If critical elements remain in "cold" zones, the design hierarchy should be reconsidered.

### b. Gaze Plot
Purpose: This simulation predicts the sequence in which a user's eye might scan across the most salient points on a visual. The route starts from the most attention-grabbing point (1) and proceeds in descending order of attention.

Interpretation:
* Positive Indicators: The route tells a logical "visual story." For instance, if the path follows a flow like "1. Headline -> 2. Product Image -> 3. Button," your design's guidance is successful.
* Areas for Improvement: A chaotic path that skips important elements suggests a problem with visual hierarchy and grouping in your design.

### c. CTA Detection and Scoring
Purpose: The algorithm finds call-to-action (CTA) buttons like "Buy," "Learn More," or "Apply" in your visual and scores their attention-grabbing potential on a scale of 0-100.

Interpretation:
* Score > 75 (Very Good): Your button is clear both textually and visually dominant.
* Score 50-74 (Good): Your button is functional but could be made more prominent.
* Score < 50 (Weak): Your button is either visually subdued or its text doesn't imply an action, making it difficult for users to notice.

## 2. System's Technical Architecture and Operation

This section details the technologies and data flow that power the application from an engineering perspective.

### Technology Stack
* Backend Framework: Python 3.10, Flask
* Image Processing: OpenCV-Python 4.x
* Optical Character Recognition (OCR): Pytesseract
* Data Visualization: Matplotlib
* Frontend: HTML5, CSS3, JavaScript
* Containerization: Docker, Docker Compose

### Data Flow and System Architecture
The application uses the Post-Redirect-Get (PRG) design pattern for a stable and secure user experience.

1.  File Upload (POST): When a user uploads a file, it is sent as `multipart/form-data` to the relevant `/upload_...` endpoint. Flask sanitizes the filename using `werkzeug.utils.secure_filename` and temporarily writes the file to the `/static/uploads` directory.

2.  Triggering the Analysis Process:
    * For Videos: The `process_video` function reads the video using `cv2.VideoCapture`. It calculates the absolute difference (`cv2.absdiff`) between frames at specific intervals (`SAMPLING_INTERVAL_SECONDS`). If this difference exceeds a certain threshold (`CHANGE_THRESHOLD`), the current frame is considered a "Key Frame".
    * For Images: The `perform_analysis` function is called directly.

3.  Core Analysis (perform_analysis): Each image (or key frame) is passed through the `generate_heatmap`, `generate_focus_map`, `generate_gaze_plot`, and `score_button_candidates` sub-functions within this central function. The output of each function (image files) is written to the `/static/outputs` directory.

4.  Data Persistence and Redirection: All analysis results are collected in a Python dictionary and stored in Flask's `session` object. The server then redirects the user to a new URL for displaying the results using `redirect(url_for('...'))`.

5.  Displaying Results (GET): The user's browser makes a standard `GET` request to this new URL. The corresponding Flask route retrieves the result data from the `session` and dynamically renders the HTML page using `render_template`.

## 3. Scientific Foundations of the Analyses

The analysis modules are based on academic principles in computer vision and cognitive psychology.

* Heatmap (Saliency): This module models the "bottom-up attention" mechanism of the human visual system, based on the foundational "Saliency-Based Visual Attention" model by Itti, Koch, and Niebur (1998).
* Gaze Plot: It simulates the "saccade" (rapid jumps) and "fixation" (short pauses) movements of the human eye by identifying high-density areas in the saliency map.
* CTA Detection: It is based on Don Norman's concept of "Affordance." The design of an element should imply how it is to be used (e.g., that it is "clickable"). The algorithm identifies this affordance by combining semantic content, visual distinctiveness, and geometric form.

## 4. Installation and Setup

You must have Git, Docker, and Docker Compose installed to run the project.

1.  Clone the Repository:
    ```bash
    git clone [https://github.com/AhmetKorkmazMe/odak-projesi.git](https://github.com/AhmetKorkmazMe/odak-projesi.git)
    ```
2.  Navigate to the Project Directory:
    ```bash
    cd odak-projesi
    ```
3.  Launch the Application:
    ```bash
    docker-compose up -d
    ```
4.  Access:
    Once complete, access the web application at `http://localhost` or `http://your_server_ip_address`.

## 5. Bibliography and References

The algorithms and methodologies used in this project are based on the following foundational scientific works and technologies.

* Itti, L., Koch, C., & Niebur, E. (1998). *A model of saliency-based visual attention for rapid scene analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence.
* Hou, X., & Zhang, L. (2007). *Saliency detection: A spectral residual approach.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
* Nielsen, J. (2006). *F-Shaped Pattern For Reading Web Content.* Nielsen Norman Group.
* Norman, D. (2013). *The Design of Everyday Things: Revised and Expanded Edition.* Basic Books.
* Google Tesseract OCR Engine & OpenCV Library.

</details>
