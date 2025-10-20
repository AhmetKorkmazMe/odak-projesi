import os
import cv2
import numpy as np
import matplotlib
import datetime
import logging
from io import BytesIO
from PIL import Image as PillowImage
import pytesseract
import secrets
from flask import Flask, render_template, request, url_for, redirect, session
from werkzeug.utils import secure_filename

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Konfigürasyon ---
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
ICON_TEMPLATE_FOLDER = 'icon_templates'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'webm', 'mov'}

app = Flask(__name__)
app.config.from_mapping(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    OUTPUT_FOLDER=OUTPUT_FOLDER,
    MAX_CONTENT_LENGTH=50 * 1024 * 1024
)
app.secret_key = secrets.token_hex(16)

for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], ICON_TEMPLATE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Yardımcı Fonksiyonlar ---
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Resim yüklenemedi: {path}")
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Analiz Fonksiyonları ---
def generate_heatmap(img, output_path):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(img)
    if not success or saliency_map is None: saliency_map = np.zeros(img.shape[:2], dtype=np.uint8)
    saliency_map_norm = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(saliency_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(output_path, overlay)
    return overlay, saliency_map_norm

def generate_focus_map(img, attn_map, output_path):
    points = []; map_copy = cv2.GaussianBlur(attn_map, (45, 45), 0)
    for _ in range(7):
        _, max_val, _, max_loc = cv2.minMaxLoc(map_copy)
        if max_val < 40: break
        points.append(max_loc); cv2.circle(map_copy, max_loc, radius=120, color=(0), thickness=-1)
    h, w, _ = img.shape
    spotlight_mask = np.zeros((h, w), dtype=np.uint8)
    if points:
        for (x, y) in points: cv2.circle(spotlight_mask, (x, y), radius=150, color=(255), thickness=-1)
    spotlight_mask_blurred = cv2.GaussianBlur(spotlight_mask, (181, 181), 0)
    spotlight_mask_final = cv2.cvtColor(spotlight_mask_blurred, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    focus_map = (img.astype(np.float32) * spotlight_mask_final).astype(np.uint8)
    cv2.imwrite(output_path, focus_map)
    return focus_map

def generate_gaze_plot(img, attn_map, output_path):
    gaze_img = img.copy(); h, w, _ = img.shape; points = []
    COLOR_RED, COLOR_YELLOW, COLOR_GREEN = (0, 0, 255), (0, 255, 255), (0, 255, 0)
    map_copy = cv2.GaussianBlur(attn_map, (45, 45), 0)
    for _ in range(10):
        _, max_val, _, max_loc = cv2.minMaxLoc(map_copy)
        if max_val < 20: break
        points.append({'pos': max_loc}); cv2.circle(map_copy, max_loc, radius=100, color=(0), thickness=-1)
    if not points: cv2.imwrite(output_path, gaze_img); return points
    for i, p in enumerate(points):
        radius = int(35 - (i * 2.5))
        p.update({'radius': max(15, radius), 'font_scale': max(0.6, max(15, radius) / 35.0), 'color': COLOR_RED if i < 3 else (COLOR_YELLOW if i < 7 else COLOR_GREEN)})
    for i in range(1, len(points)):
        p1, p2, r2, line_color = points[i-1]['pos'], points[i]['pos'], points[i]['radius'], points[i]['color']
        v = np.array(p1) - np.array(p2); dist = np.linalg.norm(v)
        if dist > r2:
            v_norm = v / dist; edge_point = np.array(p2) + v_norm * r2
            cv2.line(gaze_img, p1, (int(edge_point[0]), int(edge_point[1])), line_color, 3)
    for i, p_data in enumerate(points):
        x, y, radius, font_scale, current_color = p_data['pos'][0], p_data['pos'][1], p_data['radius'], p_data['font_scale'], p_data['color']
        safe_x, safe_y = np.clip(x, radius, w - radius), np.clip(y, radius, h - radius)
        overlay = gaze_img.copy(); cv2.circle(overlay, (safe_x, safe_y), radius, current_color, -1)
        gaze_img = cv2.addWeighted(overlay, 0.6, gaze_img, 0.4, 0)
        text = str(i + 1); font, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x, text_y = safe_x - text_size[0] // 2, safe_y + text_size[1] // 2
        cv2.putText(gaze_img, text, (text_x, text_y), font, font_scale, (0,0,0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(gaze_img, text, (text_x, text_y), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
    cv2.imwrite(output_path, gaze_img); return points

# --- EN GELİŞMİŞ VE NİHAİ CTA FONKSİYONU ---
def score_button_candidates(img, attn_map):
    gray = to_gray(img); img_h, img_w = img.shape[:2]
    CTA_KEYWORDS = ['satın al', 'sepete ekle', 'hemen al', 'sipariş ver', 'teklif al', 'kayıt ol', 'üye ol', 'giriş yap', 'başvur', 'incele', 'keşfet', 'sorgula', 'devamı', 'daha fazla', 'bilgi al', 'tümünü gör', 'buy now', 'add to cart', 'shop now', 'sign up', 'register', 'login', 'learn more', 'read more', 'discover', 'explore', 'get started', 'altyapı sorgula', 'contact us', 'detaylı incele']
    ACTION_VERBS = ['al', 'ekle', 'ver', 'ol', 'yap', 'başvur', 'incele', 'keşfet', 'sorgula', 'gör', 'tıkla', 'başla', 'izle', 'dinle', 'buy', 'add', 'shop', 'sign', 'register', 'login', 'learn', 'read', 'discover', 'explore', 'get', 'watch', 'listen']
    candidates = {}
    
    # 1. Yöntem: Metin Bölgeleri (OCR)
    try:
        data = pytesseract.image_to_data(img, lang='tur+eng', output_type=pytesseract.Output.DICT, config='--psm 11')
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 40 and len(data['text'][i].strip()) > 1:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                box = (max(0, x - 10), max(0, y - 5), min(img_w - x + 10, w + 20), min(img_h - y + 5, h + 10))
                candidates[box] = data['text'][i].strip().lower()
    except Exception as e: logging.error(f"CTA/OCR Adım 1'de hata: {e}")

    # 2. Yöntem: Geometrik Adaylar (Kenar Tespiti)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        box = (x, y, w, h)
        if box not in candidates: candidates[box] = ''

    # 3. Adım: Tüm Adayları Filtrele ve Puanla
    scored_candidates = []
    for box, text in candidates.items():
        x, y, w, h = box
        if not (img_w * 0.02 < w < img_w * 0.7 and img_h * 0.02 < h < img_h * 0.25): continue
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (1.1 < aspect_ratio < 15.0): continue

        full_text = text
        if not full_text:
            try: full_text = pytesseract.image_to_string(gray[y:y+h, x:x+w], lang='tur+eng', config='--psm 7').lower()
            except: full_text = ''
        
        keyword_score = 100 if any(keyword in full_text for keyword in CTA_KEYWORDS) else 0
        action_verb_score = 100 if any(verb in full_text.split() for verb in ACTION_VERBS) else 0
        attention_score = np.mean(attn_map[y:y+h, x:x+w]) if attn_map[y:y+h, x:x+w].size > 0 else 0
        
        headline_penalty = 0
        if keyword_score == 0 and action_verb_score == 0 and (w * h > (img_w * img_h * 0.05)) and (y < img_h * 0.35):
            headline_penalty = 400

        total_score = (keyword_score * 7) + (action_verb_score * 5) + (attention_score * 1.5) - headline_penalty

        if total_score > 120:
            scored_candidates.append({'box': box, 'score': total_score, 'has_keyword': keyword_score > 0 or action_verb_score > 0})

    if not scored_candidates: return [], 0
    
    # 4. Adım: En iyi adayları seç (Non-Maximum Suppression)
    scored_candidates.sort(key=lambda c: c['score'], reverse=True)
    unique_candidates = []
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]); yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA); boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0

    for cand in scored_candidates:
        is_unique = True
        for existing in unique_candidates:
            if calculate_iou(cand['box'], existing['box']) > 0.4:
                is_unique = False; break
        if is_unique:
            unique_candidates.append(cand)
        if len(unique_candidates) >= 5: break

    if not unique_candidates: return [], 0
    
    # 5. Adım: NİHAİ FİLTRELEME (Mutlak Anahtar Kelime Önceliği)
    keyword_ctas = [c for c in unique_candidates if c['has_keyword']]
    if keyword_ctas:
        final_candidates = keyword_ctas
        best_cta_score = final_candidates[0]['score']
        final_score = min(100, 55 + (len(final_candidates) * 10) + (best_cta_score / 1500 * 25))
    else:
        final_candidates = unique_candidates[:1]
        best_cta_score = final_candidates[0]['score']
        final_score = min(40, (best_cta_score / 1500 * 40))
    
    return final_candidates, round(final_score)

def draw_cta_box(img, cta_boxes, output_path):
    cta_img = img.copy()
    if cta_boxes:
        for i, box_info in enumerate(cta_boxes):
            box = box_info['box']
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            thickness = 4 if i == 0 else 2
            cv2.rectangle(cta_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, thickness)
    cv2.imwrite(output_path, cta_img); return cta_img

def calculate_scores(saliency_map):
    h, w = saliency_map.shape
    if h * w == 0: return {'visibility': 0, 'focus': 0, 'balanced': 0}
    total_attention = np.sum(saliency_map)
    visibility_score = total_attention / (h * w * 255) * 100 if total_attention > 0 else 0
    top_10_percent_attention = np.sum(np.sort(saliency_map.flatten())[-int(0.1 * h * w):])
    focus_score = top_10_percent_attention / total_attention * 100 if total_attention > 0 else 0
    return {'visibility': round(visibility_score, 2), 'focus': round(focus_score, 2), 'balanced': round(100 - focus_score, 2)}

def create_interpretation_table(scores):
    interpretations = {
        'visibility': {
            'Çok İyi': 'Harika! Görselinizdeki ana unsurlar (logo, başlık, ürün vb.) anında fark ediliyor. Kullanıcıların dikkatini istediğiniz noktalara başarıyla çekiyorsunuz.',
            'İyi': 'Görseliniz genel olarak dikkat çekici. Ana mesajınız büyük ihtimalle alınıyor. Dikkatin daha da artırılması için en önemli unsur ile arka plan arasındaki kontrastı güçlendirmeyi düşünebilirsiniz.',
            'Orta': 'Tasarımınızdaki bazı önemli elementler, diğer daha az önemli bölgelerin gölgesinde kalıyor olabilir. Görsel hiyerarşiyi (renk, boyut, boşluk kullanımı) gözden geçirerek ana mesajınızı daha belirgin hale getirin.',
            'Zayıf': 'Kritik! Ana mesajınız veya markanız, görsel karmaşa içinde kayboluyor. Kullanıcıların neye odaklanması gerektiğini anlaması zor. Tasarımı sadeleştirin ve en önemli tek bir unsuru öne çıkarın.'
        },
        'focus': {
            'Çok İyi': 'Mükemmel! Kullanıcı dikkati, tam olarak istediğiniz birkaç kilit noktada toplanıyor. Bu, mesajınızın net ve akılda kalıcı olmasını sağlar.',
            'İyi': 'Dikkat, büyük ölçüde ana unsurlar üzerinde. Ancak, odağı daha da keskinleştirmek için ikincil elementlerin rengini biraz daha soluklaştırabilir veya boyutlarını küçültebilirsiniz.',
            'Orta': 'Dikkat, görselin birkaç farklı noktasına dağılmış durumda. Bu durum, kullanıcının kafasını karıştırabilir. En önemli tek bir odak noktası belirleyin ve diğer her şeyi onu destekleyecek şekilde düzenleyin.',
            'Zayıf': 'Kritik! Tasarımınızda net bir odak noktası yok. Kullanıcının gözü görsel üzerinde başıboş geziniyor ve ana mesajı kaçırıyor. "Her şey önemliyse, hiçbir şey önemli değildir" ilkesini hatırlayın.'
        },
        'balanced': {
            'Çok İyi': 'İdeal denge! Tasarımınız hem belirli odak noktalarına sahip hem de genel olarak estetik bir bütünlük sunuyor. Kullanıcı hem ana mesajı alıyor hem de görselden keyif alıyor.',
            'İyi': 'İyi bir denge yakalanmış. Tasarım ne çok dağınık ne de tek bir noktaya sıkışmış. Bu, genellikle okunabilir ve taranabilir tasarımlarda görülen iyi bir durumdur.',
            'Orta': 'Tasarımınız ya tek bir noktaya aşırı odaklanmış (diğer her şey görünmez kalıyor) ya da dikkat çok fazla dağılmış. Odaklanma ve Denge arasında bir optimizasyon yapmanız gerekiyor.',
            'Zayıf': 'Denge sorunu var. Büyük ihtimalle ya tüm dikkat tek bir noktada (skor > 75) ya da tamamen dağınık (skor < 25). Tasarımınızın genel akışını ve kompozisyonunu yeniden değerlendirin.'
        },
        'cta': {
            'Çok İyi': 'Mükemmel CTA! Butonunuz hem metin olarak net bir eylem içeriyor hem de görsel olarak (konum, renk, boyut) son derece dikkat çekici. Dönüşüm için optimize edilmiş.',
            'İyi': 'Butonunuz fark ediliyor ve işlevini yerine getiriyor. Etkisini daha da artırmak için rengini çevresindeki elementlerle daha kontrastlı hale getirmeyi veya boyutunu %10-15 oranında büyütmeyi deneyebilirsiniz.',
            'Orta': 'Butonunuz ya yeterince dikkat çekmiyor ya da metni ("Daha Fazla") bir eylem içermediği için kullanıcıyı harekete geçirmekte zorlanabilir. Daha net eylem fiilleri ("Hemen Keşfet", "Teklif Al") kullanın ve görsel olarak belirginleştirin.',
            'Zayıf': 'Kritik! Eyleme çağrı butonunuz büyük ihtimalle gözden kaçıyor. Arka planla aynı renkte olabilir, çok küçük olabilir veya alakasız bir yerde konumlandırılmış olabilir. Bu, potansiyel müşteri kaybı demektir. Acilen yeniden tasarlayın.'
        }
    }
    table = []
    score_map = {'visibility': 'Görünürlük', 'focus': 'Odaklanma', 'balanced': 'Denge', 'cta': 'CTA Etkisi'}

    for key, name in score_map.items():
        score_val = round(scores.get(key, 0))
        if score_val >= 75: badge, ref = 'bg-success', 'Çok İyi'
        elif score_val >= 50: badge, ref = 'bg-primary', 'İyi'
        elif score_val >= 25: badge, ref = 'bg-warning text-dark', 'Orta'
        else: badge, ref = 'bg-danger', 'Zayıf'
        
        interpretation = interpretations[key][ref]
        table.append({'name': name, 'score': score_val, 'badge_class': badge, 'reference': ref, 'interpretation': interpretation})
        
    return table

TEXT_COLOR = '#e0e0e0'; GRID_COLOR = '#4a4a5e'; PRIMARY_COLOR = '#3a7bd5'
FACE_COLOR = '#1a1a2e'; BAR_COLORS = ['#28a745', '#ffc107', '#17a2b8', '#dc3545']
def generate_bar_chart(scores, output_path):
    labels = ['Görünürlük', 'Odaklanma', 'Denge', 'CTA Etkisi']; values = [scores.get(k, 0) for k in ['visibility', 'focus', 'balanced', 'cta']]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=FACE_COLOR); ax.set_facecolor(FACE_COLOR)
    ax.bar(labels, values, color=BAR_COLORS); ax.set_ylabel('Skor (0-100)', color=TEXT_COLOR)
    ax.set_title('Metrik Skor Dağılımı', color=TEXT_COLOR, pad=20); ax.set_ylim(0, 100)
    ax.tick_params(axis='x', colors=TEXT_COLOR); ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=GRID_COLOR)
    for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_color(GRID_COLOR)
    plt.tight_layout(); plt.savefig(output_path, facecolor=FACE_COLOR); plt.close(fig)
def generate_radar_chart(scores, output_path):
    labels = np.array(['Görünürlük', 'Odaklanma', 'Denge', 'CTA Etkisi']); stats = np.array([scores.get(k, 0) for k in ['visibility', 'focus', 'balanced', 'cta']])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]])); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor=FACE_COLOR)
    ax.set_facecolor(FACE_COLOR); ax.fill(angles, stats, color=PRIMARY_COLOR, alpha=0.4)
    ax.plot(angles, stats, color=PRIMARY_COLOR, linewidth=2); ax.set_yticklabels([])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color=TEXT_COLOR, fontsize=12)
    ax.spines['polar'].set_color(GRID_COLOR); plt.savefig(output_path, facecolor=FACE_COLOR); plt.close(fig)
def generate_timeline_chart(video_results, output_path):
    timestamps = [r['timestamp'] for r in video_results]; visibility = [r['scores']['visibility'] for r in video_results]
    focus = [r['scores']['focus'] for r in video_results]; cta = [r['scores']['cta'] for r in video_results]
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=FACE_COLOR); ax.set_facecolor(FACE_COLOR)
    ax.plot(timestamps, visibility, marker='o', linestyle='-', label='Görünürlük')
    ax.plot(timestamps, focus, marker='o', linestyle='-', label='Odaklanma')
    ax.plot(timestamps, cta, marker='o', linestyle='-', label='CTA Etkisi')
    ax.set_title('Video Boyunca Skorların Değişimi', color=TEXT_COLOR, pad=20)
    ax.set_xlabel('Zaman (Saniye)', color=TEXT_COLOR); ax.set_ylabel('Skor (0-100)', color=TEXT_COLOR)
    legend = ax.legend(facecolor=FACE_COLOR, edgecolor=GRID_COLOR); plt.setp(legend.get_texts(), color=TEXT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR); ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    for spine in ['top', 'right', 'left', 'bottom']: ax.spines[spine].set_color(GRID_COLOR)
    ax.set_ylim(0, 105); plt.tight_layout(); plt.savefig(output_path, facecolor=FACE_COLOR); plt.close(fig)

def perform_analysis(filepath, filename):
    original_img = load_image(filepath)
    file_paths = { 'original': os.path.join(app.config['UPLOAD_FOLDER'], filename), 'heatmap': os.path.join(app.config['OUTPUT_FOLDER'], f"heatmap_{filename}"), 'focus': os.path.join(app.config['OUTPUT_FOLDER'], f"focus_{filename}"), 'gaze': os.path.join(app.config['OUTPUT_FOLDER'], f"gaze_{filename}"), 'cta': os.path.join(app.config['OUTPUT_FOLDER'], f"cta_{filename}"), 'bar_chart': os.path.join(app.config['OUTPUT_FOLDER'], f"bar_{filename}.png"), 'line_chart': os.path.join(app.config['OUTPUT_FOLDER'], f"radar_{filename}.png") }
    _, saliency_map = generate_heatmap(original_img, file_paths['heatmap'])
    generate_focus_map(original_img, saliency_map, file_paths['focus'])
    gaze_points = generate_gaze_plot(original_img, saliency_map, file_paths['gaze'])
    cta_boxes, cta_score = score_button_candidates(original_img, saliency_map)
    draw_cta_box(original_img, cta_boxes, file_paths['cta'])
    scores = calculate_scores(saliency_map); scores['cta'] = cta_score
    generate_bar_chart(scores, file_paths['bar_chart'])
    generate_radar_chart(scores, file_paths['line_chart'])
    
    interpretation_table = create_interpretation_table(scores)
    
    return { "filename": filename, "scores": scores, "paths": file_paths, "interpretation_table": interpretation_table }

def process_video(video_path, filename_prefix):
    cap = cv2.VideoCapture(video_path);
    if not cap.isOpened(): return []
    results_list, previous_frame_gray, frame_count, key_frame_count = [], None, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS);
    if not fps or fps <= 0: fps = 30.0
    SAMPLING_INTERVAL_SECONDS = 2; frame_skip = int(fps * SAMPLING_INTERVAL_SECONDS)
    if frame_skip == 0: frame_skip = 1
    
    CHANGE_THRESHOLD = 3.0 # Daha hassas hale getirildi
    
    base_filename, _ = os.path.splitext(filename_prefix)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % frame_skip != 0 and frame_count > 1: continue
        current_frame_gray = to_gray(frame)
        
        is_key_frame = False
        if previous_frame_gray is None:
            is_key_frame = True
        else:
            diff = cv2.absdiff(previous_frame_gray, current_frame_gray)
            # Değişikliği daha iyi tespit etmek için piksellerin ne kadar değiştiğine bakıyoruz
            non_zero_count = np.count_nonzero(diff > 30) # Eşik 30
            change_percentage = (non_zero_count / diff.size) * 100
            if change_percentage > CHANGE_THRESHOLD:
                is_key_frame = True

        if is_key_frame:
            key_frame_count += 1; timestamp = round(frame_count / fps, 2)
            key_frame_filename = f"{base_filename}_keyframe_{key_frame_count}.jpg"
            key_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], key_frame_filename)
            cv2.imwrite(key_frame_path, frame)
            analysis_result = perform_analysis(key_frame_path, key_frame_filename)
            analysis_result['timestamp'] = timestamp
            results_list.append(analysis_result)
        
        previous_frame_gray = current_frame_gray
    cap.release(); return results_list

def cleanup_files(filename_or_id):
    try:
        session.pop('results', None); session.pop('video_results', None)
        session.pop('original_video_filename', None)
        logging.info(f"Oturum ve ilişkili geçici veriler temizlendi: {filename_or_id}")
    except Exception as e:
        logging.error(f"Oturum temizlenirken hata: {e}")

# --- FLASK ROTALARI ---
@app.route("/")
def index(): return render_template("index.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/cleanup_and_home/<path:filename>")
def cleanup_and_home(filename):
    cleanup_files(secure_filename(filename))
    return redirect(url_for('index'))

@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get('file')
    if not file or not file.filename or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS): return "Dosya seçilmedi veya geçersiz format", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        results = perform_analysis(filepath, filename)
        session['results'] = results
        return redirect(url_for('show_image_results'))
    except Exception as e:
        logging.error(f"Görsel işlenirken hata oluştu: {e}", exc_info=True)
        return "Görsel işlenirken bir hata oluştu.", 500

@app.route("/results/image")
def show_image_results():
    results = session.get('results')
    if not results: return redirect(url_for('index'))
    filename = results['filename']
    template_data = {"original_filename": filename, "interpretation_table": results['interpretation_table'], "original_url": url_for('static', filename=f'uploads/{filename}'), "heatmap_url": url_for('static', filename=f"outputs/heatmap_{filename}"), "focus_url": url_for('static', filename=f"outputs/focus_{filename}"), "gaze_url": url_for('static', filename=f"outputs/gaze_{filename}"), "cta_url": url_for('static', filename=f"outputs/cta_{filename}"), "bar_chart_url": url_for('static', filename=f"outputs/bar_{filename}.png"), "line_chart_url": url_for('static', filename=f"outputs/radar_{filename}.png")}
    return render_template("result.html", **template_data)

@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get('file')
    if not file or not file.filename or not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS): return "Dosya seçilmedi veya geçersiz format", 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        video_results = process_video(filepath, filename)
        session['video_results'] = video_results
        session['original_video_filename'] = filename
        return redirect(url_for('show_video_results'))
    except Exception as e:
        logging.error(f"Video işlenirken hata oluştu: {e}", exc_info=True)
        return "Video işlenirken bir hata oluştu.", 500

@app.route("/results/video")
def show_video_results():
    video_results = session.get('video_results'); original_filename = session.get('original_video_filename')
    if video_results is None or not original_filename: return redirect(url_for('index'))
    for result in video_results:
        fname = result['filename']
        result['urls'] = {"original": url_for('static', filename=f'uploads/{fname}'), "heatmap": url_for('static', filename=f'outputs/heatmap_{fname}'), "focus": url_for('static', filename=f'outputs/focus_{fname}'), "gaze": url_for('static', filename=f'outputs/gaze_{fname}'), "cta": url_for('static', filename=f'outputs/cta_{fname}'), "bar_chart": url_for('static', filename=f'outputs/bar_{fname}.png'), "line_chart": url_for('static', filename=f'outputs/radar_{fname}.png')}
    timeline_chart_url = ""
    if video_results:
        timeline_chart_path = os.path.join(app.config['OUTPUT_FOLDER'], f"timeline_{original_filename}.png")
        generate_timeline_chart(video_results, timeline_chart_path)
        timeline_chart_url = url_for('static', filename=f'outputs/timeline_{original_filename}.png')
    return render_template('video_result.html', video_results=video_results, original_filename=original_filename, timeline_chart_url=timeline_chart_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
