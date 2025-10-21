import os
import cv2
import numpy as np
import pytesseract # Yeni import
from PIL import Image # Yeni import (Tesseract ile daha iyi çalışır)
from datetime import datetime # Zaman damgası için yeni import

# --- Yeni: CTA Anahtar Kelime Listesi (Genişletilebilir) ---
CTA_KEYWORDS = [
    # Türkçe
    "başla", "al", "yap", "oluştur", "deneyin", "incele", "sorgula", "kaydol", "giriş", "devam",
    "et", "gönder", "satın", "ekle", "üye", "sepete", "sipariş", "ara", "başvur", "izle", "dinle",
    "tıkla", "keşfet", "ücretsiz", "randevu", "hesapla", "kullan", "katıl", "ziyaret", "bilgi", "detay", "tamam",
    # İngilizce
    "submit", "apply", "buy", "learn", "more", "start", "free", "trial", "demo", "shop", "now",
    "get", "sign", "up", "in", "join", "view", "watch", "download", "request", "contact", "add", "cart",
    "register", "explore", "discover", "click", "visit", "go", "continue", "checkout", "book", "order", "accept", "ok"
]

# --- Mevcut Yardımcı Fonksiyonlar (Değişiklik Yok) ---
def _percentile_mask(arr, p=80):
    t = np.percentile(arr, p)
    return (arr >= t).astype(np.uint8)

def _nms_peaks(sal, max_points=8, min_dist=40):
    sal_blur = cv2.GaussianBlur(sal, (0,0), 2)
    pts = []
    sal_copy = sal_blur.copy()
    for _ in range(max_points):
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sal_copy)
        if maxVal < 10: break
        pts.append(maxLoc)
        cv2.circle(sal_copy, maxLoc, min_dist, 0, -1)
    return pts # (x,y)

def _draw_focus_overlay(img, sal_norm, top_p=80):
    mask = _percentile_mask(sal_norm, p=top_p)
    mask3 = cv2.merge([mask, mask, mask])
    dimmed = (img * 0.6).astype(np.uint8)
    heat = cv2.applyColorMap(sal_norm, cv2.COLORMAP_JET)
    overlay = np.where(mask3==1, cv2.addWeighted(img, 0.4, heat, 0.6, 0), dimmed)
    return overlay, mask

def _draw_gaze_plot(img, points):
    out = img.copy()
    for i,(x,y) in enumerate(points, start=1):
        cv2.circle(out, (x,y), 16, (0,165,255), 3)
        cv2.circle(out, (x,y), 4,  (0,0,255), -1)
        cv2.putText(out, str(i), (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    return out

# --- TAMAMEN YENİLENMİŞ: Görsel CTA Adaylarını Bulan Fonksiyon (Morfolojik Yaklaşım) ---
def _find_cta_candidates(img, min_area_ratio=0.0003, aspect_ratio_range=(0.3, 12.0)):
    """Morfolojik işlemleri kullanarak buton olabilecek belirgin renkli bölgeleri bulur."""
    candidates = []
    h, w, _ = img.shape
    min_area = h * w * min_area_ratio

    # Renk segmentasyonu için HSV renk uzayına geç
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Doygunluk (Saturation) kanalını kullanarak renkli alanları bulmaya çalış
    # Düşük doygunluk genellikle gri/beyaz/siyah alanları gösterir
    saturation = hsv[:, :, 1]
    # Yüksek doygunluğa sahip alanları (renkli) seçmek için bir eşik değeri belirle
    # Bu eşik değeri, arka planın ne kadar renkli olduğuna bağlı olarak ayarlanabilir
    _, sat_thresh = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY) # 50 eşiği deneme amaçlıdır

    # Gürültüyü azaltmak ve bölgeleri birleştirmek için morfolojik işlemler
    # Açma (Opening): Küçük gürültüleri kaldırır
    # Kapama (Closing): Bölgelerdeki küçük delikleri doldurur ve ayrık bölgeleri birleştirir
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(sat_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3) # Kapamayı daha güçlü yapalım

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, wc, hc = cv2.boundingRect(c)
        area = wc * hc
        if area < min_area or hc == 0 or wc == 0: continue

        ar = wc / float(hc)

        # Sadece en boy oranı makul olanları aday olarak al
        if aspect_ratio_range[0] <= ar <= aspect_ratio_range[1]:
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + wc + padding)
            y2 = min(h, y + hc + padding)
            candidates.append((x1, y1, x2 - x1, y2 - y1))

    # Alternatif: Sadece doygunluk yerine, belirli renk aralıklarını (örn. mavi, turuncu)
    # cv2.inRange ile maskeleyip bu maskeler üzerinde kontur bulma da denenebilir.
    # Şimdilik bu daha genel doygunluk yaklaşımını deneyelim.
    return candidates

# --- Yeni: Adayları OCR ile Filtreleyen Fonksiyon (Teşhis Printi Hala İçinde) ---
def _filter_candidates_with_ocr(img, candidates):
    """Aday kutuları içindeki metni okuyarak CTA anahtar kelimesi içerip içermediğini kontrol eder."""
    confirmed_ctas = []
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for (x, y, wc, hc) in candidates:
        roi_gray = gray[y:y+hc, x:x+wc]
        if roi_gray.size == 0 or wc < 10 or hc < 10: continue

        try:
            # OCR için Adaptive Thresholding deneyelim, farklı aydınlatmalara daha iyi uyum sağlayabilir
            roi_thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
            # _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Eski yöntem
        except Exception:
             roi_thresh = roi_gray # Hata olursa griyi kullan

        try:
            custom_config = r'--oem 3 --psm 6 -l tur+eng'
            text = pytesseract.image_to_string(Image.fromarray(roi_thresh), config=custom_config).strip()

            if not text: continue

            words = text.lower().split()
            found_keyword = False
            matched_keyword = None
            cleaned_words_list = []

            for word in words:
                cleaned_word = ''.join(filter(str.isalnum, word))
                cleaned_words_list.append(cleaned_word)
                if cleaned_word and cleaned_word in CTA_KEYWORDS:
                    found_keyword = True
                    matched_keyword = cleaned_word
                    break

            # !!!!! TEŞHİS PRINT SATIRI !!!!!
            print(f"[DEBUG OCR] Box({x},{y},{wc},{hc}) | Text: '{text}' | Cleaned: {cleaned_words_list} | Keyword Found: {found_keyword} | Match: {matched_keyword}")
            # !!!!! TEŞHİS PRINT SATIRI !!!!!

            if found_keyword:
                confirmed_ctas.append((x, y, wc, hc))

        except Exception as e:
            print(f"OCR Error processing box ({x},{y},{wc},{hc}): {e}")
            continue

    return confirmed_ctas

# --- Ana `analyze` Fonksiyonu Güncellemesi (Geri Kalanı Aynı) ---
def analyze(image_path, out_dir):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Görsel okunamadı.")
    h_orig, w_orig, _ = img.shape

    sal_instance = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_f = sal_instance.computeSaliency(img)
    if not ok: raise RuntimeError("Saliency üretilemedi.")
    sal_u8 = (sal_f * 255).astype("uint8")
    if sal_u8.shape[:2] != img.shape[:2]:
        sal_u8 = cv2.resize(sal_u8, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    sal_u8 = cv2.normalize(sal_u8, None, 0, 255, cv2.NORM_MINMAX)

    focus_img, mask = _draw_focus_overlay(img, sal_u8, top_p=80)
    points = _nms_peaks(sal_u8, max_points=8, min_dist=40)
    gaze_img = _draw_gaze_plot((img * 0.7 + np.dstack([sal_u8] * 3) * 0.3).astype(np.uint8), points)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_base = f"{timestamp}_{base}"

    focus_name = f"focus_{unique_base}.jpg"
    gaze_name = f"gaze_{unique_base}.jpg"
    cv2.imwrite(os.path.join(out_dir, focus_name), focus_img)
    cv2.imwrite(os.path.join(out_dir, gaze_name), gaze_img)

    visibility = float(mask.sum()) / mask.size * 100.0 if mask.size > 0 else 0.0
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    area = mask.sum()
    if num > 1 and area > 0:
        largest = max((labels == i).sum() for i in range(1, num))
        concentration = (largest / area * 100.0)
    else:
        concentration = 100.0 if area > 0 else 0.0

    h, w = mask.shape
    ch1, ch2 = int(h * 0.3), int(h * 0.7)
    cw1, cw2 = int(w * 0.3), int(w * 0.7)
    center_roi = sal_u8[ch1:ch2, cw1:cw2]
    ctr_mean = center_roi.mean() if center_roi.size > 0 else 0.0
    all_mean = sal_u8.mean() if sal_u8.size > 0 else 0.0
    center_bias = min(100.0, max(0.0, (ctr_mean / (all_mean + 1e-6)) * 50 + 50))

    visual_candidates = _find_cta_candidates(img)
    # print(f"[DEBUG VISUAL] Found {len(visual_candidates)} visual candidates.") # Bu satırı aktif edebilirsiniz
    confirmed_ctas = _filter_candidates_with_ocr(img, visual_candidates)

    best_cta_box = None
    best_cta_score = -1.0

    if confirmed_ctas:
        max_saliency_score = -1.0
        for (x, y, W, H) in confirmed_ctas:
            y_end = min(h_orig, y + H)
            x_end = min(w_orig, x + W)
            region = sal_u8[max(0, y):y_end, max(0, x):x_end]
            if region.size > 0:
                avg_saliency = float(region.mean())
                if avg_saliency > max_saliency_score:
                    max_saliency_score = avg_saliency
                    best_cta_box = (x, y, W, H)

        if best_cta_box is not None:
             best_cta_score = (max_saliency_score / 255.0) * 100.0

    scores = {
        "attention": round(visibility, 1),
        "focus_distribution": round(concentration, 1),
        "cta_visibility": round(best_cta_score, 1) if best_cta_score >= 0 else None,
        "center_bias": round(center_bias, 1)
    }

    preview_name = None
    if best_cta_box:
        pv = focus_img.copy()
        x, y, W, H = best_cta_box
        x_end = min(w_orig, x + W)
        y_end = min(h_orig, y + H)
        cv2.rectangle(pv, (max(0,x), max(0,y)), (x_end, y_end), (36, 255, 12), 3)
        preview_name = f"cta_{unique_base}.jpg"
        cv2.imwrite(os.path.join(out_dir, preview_name), pv)

    return {
        "focus_file": focus_name,
        "gaze_file":  gaze_name,
        "cta_preview": preview_name,
        "cta_box": best_cta_box,
        "scores": scores
    }

# --- Dosya sonu ---
