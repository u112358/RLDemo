// three.js view of the pocket cube, shared by playground.html and dashboard.html.
// Requires THREE (r128) and CubeModel to be loaded first.
//
//   const view = CubeView.create(canvas, { palette: CubeModel.ENV_COLORS });
//   view.setState('wggbrymrgrwrbmbwybywmygm');
//   view.animate(action, durationMs, nextState, () => { ... });
(function (global) {
  'use strict';
  const M = global.CubeModel;
  const LAYER = [
    { axis: 'z', sign: -1, sel: p => p.z > 0.1 },   // top layer, about z
    { axis: 'x', sign: -1, sel: p => p.x > 0.1 },   // right layer, about x
    { axis: 'y', sign: 1, sel: p => p.y < -0.1 },   // front layer, about y
  ];
  const easeInOutCubic = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  function roundedRect(w, h, r) {
    const s = new THREE.Shape(), x = -w / 2, y = -h / 2;
    s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
    s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
    s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
    return s;
  }

  function create(canvas, opts) {
    opts = opts || {};
    const available = typeof THREE !== 'undefined';
    let palette = Object.assign({}, opts.palette || M.ENV_COLORS);
    let state = opts.state || M.SOLVED;
    if (!available) {
      // no library: keep the API but do nothing visual
      return {
        available: false, get state() { return state; },
        setState(s) { state = s; }, setPalette(p) { palette = Object.assign({}, p); },
        animate(a, dur, next, cb) { state = next; if (cb) cb(); }, busy: false, setView() {},
      };
    }
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputEncoding = THREE.sRGBEncoding;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    camera.up.set(0, 0, 1);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x50566a, 0.95));
    const sun = new THREE.DirectionalLight(0xffffff, 0.75); sun.position.set(3, -4, 6); scene.add(sun);
    const fill = new THREE.DirectionalLight(0xffffff, 0.25); fill.position.set(-4, 3, -2); scene.add(fill);
    const root = new THREE.Group(); scene.add(root);
    const pivot = new THREE.Group(); root.add(pivot);

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1a1d24, roughness: 0.55, metalness: 0.05 });
    const cubieGeo = new THREE.BoxGeometry(0.97, 0.97, 0.97);
    const cubies = [];
    for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) {
      const m = new THREE.Mesh(cubieGeo, bodyMat); m.position.set(x, y, z); root.add(m); cubies.push(m);
    }
    const stickerGeo = new THREE.ShapeGeometry(roundedRect(0.84, 0.84, 0.12));
    const stickers = [];
    for (let i = 0; i < 24; i++) {
      const c = M.SLOT_POS[i], n = M.SLOT_NORMAL[i];
      const mesh = new THREE.Mesh(stickerGeo, new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0 }));
      mesh.position.set(c[0] - n[0] * 0.006, c[1] - n[1] * 0.006, c[2] - n[2] * 0.006);
      if (n[2] !== 0) mesh.up.set(0, 1, 0); else mesh.up.set(0, 0, 1);
      mesh.lookAt(mesh.position.x + n[0], mesh.position.y + n[1], mesh.position.z + n[2]);
      root.add(mesh);
      stickers.push({ mesh, pos0: mesh.position.clone(), quat0: mesh.quaternion.clone() });
    }

    // camera orbit, z up
    let theta = opts.theta ?? -0.95, phi = opts.phi ?? 0.48, dist = opts.dist ?? 6.4;
    function placeCamera() {
      camera.position.set(dist * Math.cos(phi) * Math.cos(theta), dist * Math.cos(phi) * Math.sin(theta), dist * Math.sin(phi));
      camera.lookAt(0, 0, 0);
    }
    let drag = null;
    canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      theta -= (e.clientX - drag.x) * 0.008; phi += (e.clientY - drag.y) * 0.008;
      phi = Math.max(-1.45, Math.min(1.45, phi)); drag = { x: e.clientX, y: e.clientY }; placeCamera();
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('wheel', e => { e.preventDefault(); dist = Math.max(4, Math.min(11, dist + e.deltaY * 0.004)); placeCamera(); }, { passive: false });
    function resize() {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    new ResizeObserver(resize).observe(canvas);
    resize(); placeCamera();

    function paint() {
      for (let i = 0; i < 24; i++) stickers[i].mesh.material.color.set(palette[state[i]] || '#888888').convertSRGBToLinear();
    }
    function resetTransforms() {
      for (const st of stickers) { st.mesh.position.copy(st.pos0); st.mesh.quaternion.copy(st.quat0); }
      for (const c of cubies) { c.position.set(Math.sign(c.position.x) * 0.5, Math.sign(c.position.y) * 0.5, Math.sign(c.position.z) * 0.5); c.rotation.set(0, 0, 0); }
      pivot.rotation.set(0, 0, 0);
    }
    const view = {
      available: true, busy: false,
      get state() { return state; },
      setState(s) { state = s; paint(); },
      setPalette(p) { palette = Object.assign({}, p); paint(); },
      setView(t, p, d) { if (t !== undefined) theta = t; if (p !== undefined) phi = p; if (d !== undefined) dist = d; placeCamera(); },
      /** Rotate the layer of action `a` over `dur` ms, then show `next` and call `cb`. */
      animate(a, dur, next, cb) {
        const layer = LAYER[Math.floor(a / 3)], k = a % 3 + 1;
        const quarters = k === 3 ? -1 : k;
        const target = layer.sign * quarters * Math.PI / 2;
        const parts = cubies.concat(stickers.map(s => s.mesh)).filter(m => layer.sel(m.position));
        parts.forEach(m => pivot.attach(m));
        const t0 = performance.now();
        const total = reduced ? 0 : dur * (k === 2 ? 1.35 : 1);
        view.busy = true;
        function step(now) {
          const t = total ? Math.min(1, (now - t0) / total) : 1;
          pivot.rotation[layer.axis] = target * easeInOutCubic(t);
          if (t < 1) { requestAnimationFrame(step); return; }
          parts.forEach(m => root.attach(m));
          resetTransforms();
          state = next; paint();
          view.busy = false;
          if (cb) cb();
        }
        requestAnimationFrame(step);
      },
    };
    paint();
    (function loop() { renderer.render(scene, camera); requestAnimationFrame(loop); })();
    return view;
  }

  global.CubeView = { create, available: typeof THREE !== 'undefined' };
})(window);
