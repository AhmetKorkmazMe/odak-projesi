(function () {
  const img = document.getElementById('aoiImage');
  const canvas = document.getElementById('aoiCanvas');
  const btnAdd = document.getElementById('btnAdd');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const jobId = document.getElementById('job_id').value;
  const tbody = document.querySelector('#aoiTable tbody');

  let boxes = [];          // [{x,y,w,h} ...]
  let drawing = false;
  let start = null;

  function fitCanvas() {
    const r = img.getBoundingClientRect();
    canvas.width = r.width;
    canvas.height = r.height;
    canvas.style.left = img.offsetLeft + 'px';
    canvas.style.top  = img.offsetTop  + 'px';
    draw();
  }

  function imgPoint(evt) {
    const r = img.getBoundingClientRect();
    const x = Math.max(0, Math.min(evt.clientX - r.left, r.width));
    const y = Math.max(0, Math.min(evt.clientY - r.top, r.height));
    return { x, y, iw: r.width, ih: r.height };
  }

  function draw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.strokeStyle = '#0d6efd';
    ctx.fillStyle = 'rgba(13,110,253,.15)';
    ctx.lineWidth = 2;

    for (const b of boxes) {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillRect(b.x, b.y, b.w, b.h);
    }
    if (drawing && start) {
      const {x,y} = start;
      const cur = latest;
      const w = cur.x - x, h = cur.y - y;
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, y, w, h);
    }
  }

  let latest = {x:0,y:0};
  btnAdd.addEventListener('click', () => {
    canvas.style.pointerEvents = 'auto';
    drawing = false; start=null; latest={x:0,y:0};
  });

  canvas.addEventListener('mousedown', e => {
    const p = imgPoint(e);
    drawing = true; start = {x:p.x, y:p.y};
  });
  canvas.addEventListener('mousemove', e => {
    if (!drawing) return;
    const p = imgPoint(e); latest = {x:p.x, y:p.y}; draw();
  });
  canvas.addEventListener('mouseup', e => {
    if (!drawing) return;
    drawing = false;
    const p = imgPoint(e);
    const x = Math.min(start.x, p.x), y = Math.min(start.y, p.y);
    const w = Math.abs(p.x - start.x), h = Math.abs(p.y - start.y);
    if (w>10 && h>10) boxes.push({x,y,w,h, iw:p.iw, ih:p.ih});
    draw();
    canvas.style.pointerEvents = 'none';
  });

  window.addEventListener('resize', fitCanvas);
  img.addEventListener('load', fitCanvas);
  if (img.complete) fitCanvas();

  btnAnalyze.addEventListener('click', async () => {
    if (boxes.length === 0) return;

    // Görsel piksel koordinatlarına ölçekle (server gerçek boyutta çalışır)
    const naturalW = img.naturalWidth, naturalH = img.naturalHeight;
    const scaled = boxes.map(b => {
      return [
        Math.round(b.x * (naturalW / b.iw)),
        Math.round(b.y * (naturalH / b.ih)),
        Math.round(b.w * (naturalW / b.iw)),
        Math.round(b.h * (naturalH / b.ih)),
      ];
    });

    const res = await axios.post('/api/aoi', { job_id: jobId, boxes: scaled });
    // tablo
    tbody.innerHTML = '';
    res.data.rows.forEach((r, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${i+1}</td><td>[${r.box.join(', ')}]</td><td>${r.visibility}%</td><td>${r.fixations}</td><td>${r.time_to_first_fixation}</td>`;
      tbody.appendChild(tr);
    });
    // overlay ve PDF linkini güncelle
    document.getElementById('aoiImage').src = res.data.aoi_overlay_url + `?v=${Date.now()}`;
    document.querySelector('a.btn.btn-light').href = res.data.pdf_url;
    fitCanvas();
  });
})();
