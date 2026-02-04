import streamlit as st
import packaging.version
import gc  # Added for garbage collection

# ==========================================
# 0. VERSION CHECK
# ==========================================
required_version = "1.33.0"
current_version = st.__version__
if packaging.version.parse(current_version) >= packaging.version.parse("1.34.0"):
    st.warning(f"⚠️ SYSTEM WARNING: Streamlit {current_version} detected. Downgrade to 1.33.0 for visual stability.")

import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from PIL import Image
import heapq
import tempfile
import os
import time
import pandas as pd
from streamlit_drawable_canvas import st_canvas

# ==========================================
# 1. OPTIMIZED CONFIGURATION
# ==========================================
st.set_page_config(page_title="Neural Nav System", page_icon="💠", layout="wide")

# Set memory allocation config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Orbitron:wght@500;700;900&display=swap');
    .stApp { background: radial-gradient(circle at 50% 0%, #1a202c 0%, #050505 100%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 2px; color: #00f2ff; text-shadow: 0 0 15px rgba(0, 242, 255, 0.4); }
    .stMetric, div[data-testid="stDataFrame"] { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 15px; backdrop-filter: blur(10px); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); }
    div[data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; font-size: 1.8rem !important; color: #ff007f !important; text-shadow: 0 0 10px rgba(255, 0, 127, 0.5); }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    iframe { border: 1px solid #00f2ff; border-radius: 4px; box-shadow: 0 0 20px rgba(0, 242, 255, 0.15); }
    .stButton>button { background: linear-gradient(90deg, #00f2ff 0%, #0099ff 100%); color: #000; font-family: 'Orbitron', sans-serif; font-weight: bold; border: none; border-radius: 4px; transition: transform 0.2s, box-shadow 0.2s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 0 15px rgba(0, 242, 255, 0.6); color: #000; }
    header {visibility: hidden;} footer {visibility: hidden;} .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

class PlannerConfig:
    # OPTIMIZATION: Reduced width slightly to prevent OOM on <8GB VRAM
    TARGET_WIDTH = 960
    SCALE = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PATH_COLORS = [
        (0, 242, 255), (255, 0, 127), (57, 255, 20), (255, 209, 0), (255, 80, 80)
    ]
    BRUSH_COLORS = ["#00f2ff", "#ff007f", "#39ff14", "#ffd100", "#ff5050"]

    CLASS_COSTS = {
        0: 255, 1: 255, 2: 1, 3: 20, 4: 10,
        5: 5, 6: 255, 7: 255, 8: 1, 9: 255
    }

# Session State
if 'canvas_key' not in st.session_state: st.session_state.canvas_key = 0
if 'path_idx' not in st.session_state: st.session_state.path_idx = 0
if 'baked_map' not in st.session_state: st.session_state.baked_map = None
if 'original_map' not in st.session_state: st.session_state.original_map = None
if 'metrics_log' not in st.session_state: st.session_state.metrics_log = []
if 'img_dims' not in st.session_state: st.session_state.img_dims = (544, 960)

# ==========================================
# 2. CORE LOGIC (MEMORY SAFE)
# ==========================================
@st.cache_resource
def load_model(path):
    model = smp.Unet(encoder_name="resnet34", classes=10)
    ckpt = torch.load(path, map_location=PlannerConfig.DEVICE)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items(): new_sd[k.replace('module.', '')] = v
    model.load_state_dict(new_sd)
    return model.to(PlannerConfig.DEVICE).eval()

def resize_smart(pil_img, target_width):
    w, h = pil_img.size
    aspect = h / w
    new_w = target_width
    new_h = int(new_w * aspect)
    new_h = (new_h // 32) * 32
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def process_terrain(model, pil_img):
    """
    Inference with aggressive memory cleanup.
    """
    resized_pil = resize_smart(pil_img, PlannerConfig.TARGET_WIDTH)
    img = np.array(resized_pil.convert('RGB'))
    H, W = img.shape[:2]

    t0 = time.time()

    # 1. Create Tensor
    inp = (img.astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    inp_tensor = torch.from_numpy(inp.transpose(2,0,1)).float().unsqueeze(0).to(PlannerConfig.DEVICE)

    # 2. Inference
    with torch.no_grad():
        output = model(inp_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]

    inf_time = time.time() - t0

    # 3. CRITICAL: Aggressive Cleanup to release VRAM
    del inp
    del inp_tensor
    del output
    if PlannerConfig.DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Post-Process (CPU only from here)
    h_s, w_s = int(H * PlannerConfig.SCALE), int(W * PlannerConfig.SCALE)
    cost = cv2.resize(pred.astype(np.uint8), (w_s, h_s), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    for k, v in PlannerConfig.CLASS_COSTS.items(): cost[cost == k] = v

    obs = (cost >= 255).astype(np.uint8)
    cost[cv2.dilate(obs, np.ones((3,3), np.uint8)) == 1] = 255
    return cost, img, inf_time, (H, W)

class AStar:
    def __init__(self, m): self.m = m
    def h(self, a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def plan(self, s, g):
        t0 = time.time()
        q, seen, costs = [(0, 0, s)], {s: None}, {s: 0}
        nodes = 0
        while q:
            _, cur_c, curr = heapq.heappop(q)
            nodes += 1
            if curr == g: break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = curr[0]+dr, curr[1]+dc
                if 0<=nr<self.m.shape[0] and 0<=nc<self.m.shape[1] and self.m[nr,nc]<255:
                    weight = 1.414 if abs(dr)+abs(dc)==2 else 1.0
                    new_c = cur_c + weight + (self.m[nr,nc] * 0.1)
                    if (nr,nc) not in costs or new_c < costs[(nr,nc)]:
                        costs[(nr,nc)] = new_c
                        heapq.heappush(q, (new_c + self.h(g, (nr,nc)), new_c, (nr,nc)))
                        seen[(nr,nc)] = curr
        dt = time.time() - t0
        if g not in seen: return None, dt, nodes, 0
        p, c = [], g
        while c: p.append(c); c = seen[c]
        return p[::-1], dt, nodes, costs[g]

def snap_to_safe(m, node):
    r, c = node
    if not (0 <= r < m.shape[0] and 0 <= c < m.shape[1]): return None
    if m[r, c] < 255: return node
    for d in range(1, 15):
        for dr in range(-d, d+1):
            for dc in range(-d, d+1):
                nr, nc = r+dr, c+dc
                if 0<=nr<m.shape[0] and 0<=nc<m.shape[1] and m[nr,nc]<255: return (nr,nc)
    return None

# ==========================================
# 3. INTERFACE
# ==========================================
st.title("NEURAL NAV SYSTEM // V9")

with st.sidebar:
    st.header("SYSTEM INPUT")
    m_file = st.file_uploader("Model Weights (.pth)", type=['pth'])
    st.divider()
    if st.button("REBOOT SYSTEM", type="primary"):
        st.session_state.baked_map = st.session_state.original_map.copy() if st.session_state.original_map is not None else None
        st.session_state.path_idx = 0
        st.session_state.metrics_log = []
        st.session_state.canvas_key += 1
        st.rerun()

if m_file:
    with tempfile.NamedTemporaryFile(delete=False) as t:
        t.write(m_file.read()); t_p = t.name
    model = load_model(t_p)
    i_file = st.file_uploader("Terrain Data (.img)", type=['jpg','png','jpeg'])

    if i_file:
        file_id = f"{i_file.name}_{i_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            with st.spinner("INITIALIZING SEGMENTATION CORE..."):
                cost_map, disp_img, inf_time, dims = process_terrain(model, Image.open(i_file))
                st.session_state.cost_map = cost_map
                st.session_state.original_map = disp_img.copy()
                st.session_state.baked_map = disp_img.copy()
                st.session_state.last_file_id = file_id
                st.session_state.path_idx = 0
                st.session_state.inference_time = inf_time
                st.session_state.img_dims = dims
                st.session_state.metrics_log = []
                st.session_state.canvas_key += 1

        cost_map = st.session_state.cost_map
        current_map = st.session_state.baked_map
        H, W = st.session_state.img_dims
        p_idx = st.session_state.path_idx % 5
        brush_color = PlannerConfig.BRUSH_COLORS[p_idx]

        pad1, center_col, pad2 = st.columns([1, 12, 1])
        with center_col:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;">
                <span><strong>ACTIVE AGENT:</strong> <code style="color:{brush_color}">UNIT_{p_idx}</code></span>
                <span>STATUS: <span style="color:#39ff14">ONLINE</span></span>
            </div>
            """, unsafe_allow_html=True)

            canvas = st_canvas(
                fill_color=brush_color, stroke_width=0,
                background_image=Image.fromarray(current_map),
                height=H, width=W,
                drawing_mode="point", point_display_radius=6,
                update_streamlit=True,
                key=f"hud_canvas_{st.session_state.canvas_key}"
            )

        st.divider()
        st.markdown("### MISSION TELEMETRY")

        if canvas.json_data:
            pts = [o for o in canvas.json_data["objects"] if o["type"] == "circle"]
            if len(pts) == 2:
                s_raw = (int(pts[0]["top"]*PlannerConfig.SCALE), int(pts[0]["left"]*PlannerConfig.SCALE))
                g_raw = (int(pts[1]["top"]*PlannerConfig.SCALE), int(pts[1]["left"]*PlannerConfig.SCALE))

                s_node = snap_to_safe(cost_map, s_raw)
                g_node = snap_to_safe(cost_map, g_raw)

                if s_node and g_node:
                    astar = AStar(cost_map)
                    path, p_time, nodes, t_cost = astar.plan(s_node, g_node)

                    if path:
                        path_px = np.array([[int(n[1]/PlannerConfig.SCALE), int(n[0]/PlannerConfig.SCALE)] for n in path], np.int32)
                        col_bgr = PlannerConfig.PATH_COLORS[p_idx]
                        cv2.polylines(st.session_state.baked_map, [path_px.reshape(-1,1,2)], False, col_bgr, 3)
                        cv2.circle(st.session_state.baked_map, (path_px[0][0], path_px[0][1]), 6, col_bgr, -1)
                        cv2.circle(st.session_state.baked_map, (path_px[-1][0], path_px[-1][1]), 6, col_bgr, -1)

                        st.session_state.metrics_log.append({
                            "UNIT": f"#{p_idx}",
                            "LATENCY": f"{round(p_time*1000, 2)}ms",
                            "STEPS": len(path),
                            "COST": round(t_cost, 2),
                            "EFFICIENCY": round(nodes/len(path), 2)
                        })

                        st.session_state.path_idx += 1
                        st.session_state.canvas_key += 1
                        st.rerun()
                    else: st.error("TRAJECTORY ERROR: PATH OBSTRUCTED")
                else: st.error("COORDINATE ERROR: UNSAFE TERRAIN")

        m1, m2 = st.columns([2, 1])
        with m1:
            if st.session_state.metrics_log:
                st.dataframe(pd.DataFrame(st.session_state.metrics_log), use_container_width=True)
            else: st.info("Awaiting Mission Data...")

        with m2:
            st.metric("Inference Latency", f"{st.session_state.get('inference_time',0)*1000:.1f} ms")
            if st.session_state.metrics_log:
                st.metric("Traversal Cost", st.session_state.metrics_log[-1]["COST"])

        with st.expander("INTERNAL LAYERS [DEBUG]"):
            d1, d2 = st.columns(2)
            d1.image(cost_map/255.0, caption="Cost Map (Raw)", clamp=True, use_column_width=True)
            d2.image(cv2.applyColorMap(cost_map.astype(np.uint8), cv2.COLORMAP_JET), caption="Thermal Heatmap", use_column_width=True)
