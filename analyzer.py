import os, cv2, numpy as np

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
    return pts  # (x,y)

def _draw_focus_overlay(img, sal_norm, top_p=80):
    mask = _percentile_mask(sal_norm, p=top_p)
    mask3 = cv2.merge([mask, mask, mask])
    dimmed = (img * 0.6).astype(np.uint8)
    heat  = cv2.applyColorMap(sal_norm, cv2.COLORMAP_JET)
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

def _naive_cta_detect(img):
    """Dolu renkli dikdörtgen benzeri alanı dene (çok naif)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1]
    _, sat_bin = cv2.threshold(sat, 100, 255, cv2.THRESH_BINARY)
    sat_bin = cv2.morphologyEx(sat_bin, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    contours,_ = cv2.findContours(sat_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = None; best=0
    h,w = sat_bin.shape
    for c in contours:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        if area < (h*w)*0.005: continue
        ar = wc/float(hc)
        if 1.5 <= ar <= 6.0: # buton benzeri
            fill = cv2.countNonZero(sat_bin[y:y+hc, x:x+wc])/float(area)
            if fill>0.7 and area>best:
                best=area; cand=(x,y,wc,hc)
    return cand  # None or (x,y,w,h)

def analyze(image_path, out_dir, want_cta_box=None):
    img = cv2.imread(image_path)
    if img is None: raise ValueError("Görsel okunamadı.")

    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_f = sal.computeSaliency(img)
    if not ok: raise RuntimeError("Saliency üretilemedi.")
    sal_u8 = (sal_f*255).astype("uint8")
    sal_u8 = cv2.normalize(sal_u8, None, 0, 255, cv2.NORM_MINMAX)

    focus_img, mask = _draw_focus_overlay(img, sal_u8, top_p=80)
    points = _nms_peaks(sal_u8, max_points=8, min_dist=40)
    gaze_img = _draw_gaze_plot((img*0.7+np.dstack([sal_u8]*3)*0.3).astype(np.uint8), points)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(image_path)
    focus_name = f"focus_{base}"
    gaze_name  = f"gaze_{base}"
    cv2.imwrite(os.path.join(out_dir, focus_name), focus_img)
    cv2.imwrite(os.path.join(out_dir, gaze_name),  gaze_img)

    # skorlar
    visibility = float(mask.sum())/mask.size*100.0
    # kümelenme: en büyük bağlı bileşen oranı
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    largest = max((labels==i).sum() for i in range(1,num)) if num>1 else 0
    area = mask.sum()
    concentration = (largest/area*100.0) if area>0 else 0.0
    # center-bias (CTA vekili)
    h,w = mask.shape; ch1,ch2 = int(h*0.3),int(h*0.7); cw1,cw2 = int(w*0.3),int(w*0.7)
    ctr_mean = sal_u8[ch1:ch2, cw1:cw2].mean(); all_mean = sal_u8.mean()
    cta_vis = min(100.0, max(0.0, (ctr_mean/(all_mean+1e-6))*50+50))

    # CTA kutusu (opsiyonel)
    cta_box = None
    if want_cta_box and all(k in want_cta_box for k in ("x","y","w","h")):
        cta_box = (int(want_cta_box["x"]), int(want_cta_box["y"]), int(want_cta_box["w"]), int(want_cta_box["h"]))
    else:
        cta_box = _naive_cta_detect(img)

    cta_score = None
    if cta_box:
        x,y,W,H = cta_box
        region = sal_u8[max(0,y):y+H, max(0,x):x+W]
        if region.size>0:
            cta_score = float(region.mean())/255.0*100.0

    scores = {
        "attention": round(visibility,1),
        "focus_distribution": round(concentration,1),
        "cta_visibility": round(cta_score,1) if cta_score is not None else None,
        "center_bias": round(cta_vis,1)
    }

    # çizimli CTA önizleme
    preview_name = None
    if cta_box:
        pv = focus_img.copy()
        x,y,W,H = cta_box
        cv2.rectangle(pv, (x,y), (x+W,y+H), (0,255,0), 3)
        preview_name = f"cta_{base}"
        cv2.imwrite(os.path.join(out_dir, preview_name), pv)

    return {
        "focus_file": focus_name,
        "gaze_file":  gaze_name,
        "cta_preview": preview_name,
        "cta_box": cta_box,
        "scores": scores
    }
