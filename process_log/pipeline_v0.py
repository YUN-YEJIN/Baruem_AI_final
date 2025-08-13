''''tongue.zip'에서 받아와 수정중이었던 파일.'''

import os
import json
import argparse
from typing import List, Tuple, Dict
import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

Point = Tuple[float, float]

BASE_DIR = r"C:\Users\user\PycharmProjects\betterpronounciation\tongue"
IMAGES_DIR = os.path.join(BASE_DIR, "image_v0")
LANDMARKS_DIR = os.path.join(BASE_DIR, "../tongue2/results")
MAPPING_JSON = os.path.join(BASE_DIR, '../tongue2/phoneme_mapping.json')
OUTPUT_DIR = os.path.join(BASE_DIR, '../tongue2/output')

SEMIVOWEL_FILES = {
    'ㅣ_반모음': {
        'image': 'iiJ.png',
        'landmarks': 'iiJ_manual_landmarks.json'
    },
    'ㅜ_반모음': {
        'image': 'eui-uuW.png', 
        'landmarks': 'eui-uuW_manual_landmarks.json'
    }
}

DIPHTHONG_MAPPING = {
    'ㅕ_혀': ('ㅣ_혀', 'ㅣ_반모음', 'ㅓ_혀'),
    'ㅛ_혀': ('ㅣ_혀', 'ㅣ_반모음', 'ㅗ_혀'),
    'ㅠ_혀': ('ㅣ_혀', 'ㅣ_반모음', 'ㅜ_혀'),
    'ㅑ_혀': ('ㅣ_혀', 'ㅣ_반모음', 'ㅏ_혀'),
    'ㅞ_혀': ('ㅜ_혀', 'ㅜ_반모음', 'ㅔ_혀'),
    'ㅟ_혀': ('ㅜ_혀', 'ㅜ_반모음', 'ㅣ_혀'),
    'ㅝ_혀': ('ㅜ_혀', 'ㅜ_반모음', 'ㅓ_혀'),
    'ㅘ_혀': ('ㅜ_혀', 'ㅜ_반모음', 'ㅏ_혀'),
    'ㅢ_혀': ('ㅡ_혀', 'ㅜ_반모음', 'ㅣ_혀'),
    'ㅒ_혀': ('ㅣ_혀', 'ㅣ_반모음', 'ㅐ_혀'),
    'ㅖ_혀': ('ㅣ_혀', 'ㅣ_반모음', 'ㅔ_혀'),
    'ㅕ_입': ('ㅣ_입', 'ㅣ_반모음', 'ㅓ_입'),
    'ㅛ_입': ('ㅣ_입', 'ㅣ_반모음', 'ㅗ_입'),
    'ㅠ_입': ('ㅣ_입', 'ㅣ_반모음', 'ㅜ_입'),
    'ㅑ_입': ('ㅣ_입', 'ㅣ_반모음', 'ㅏ_입'),
}

def create_semivowel_entry(semivowel_type: str, target_type: str) -> Dict:
    if semivowel_type not in SEMIVOWEL_FILES:
        raise KeyError(f"Semivowel type {semivowel_type} not found")
    
    files = SEMIVOWEL_FILES[semivowel_type]
    return {
        'label': f"{semivowel_type}_{target_type}",
        'image': files['image'], 
        'landmarks': files['landmarks']
    }

def is_diphthong(phoneme: str) -> bool:
    return phoneme in DIPHTHONG_MAPPING

def expand_diphthong(phoneme: str) -> List[str]:
    if is_diphthong(phoneme):
        start, semivowel_type, end = DIPHTHONG_MAPPING[phoneme]
        target_type = phoneme.split('_')[-1]
        semivowel_full = f"{semivowel_type}_{target_type}"
        return [start, semivowel_full, end]
    else:
        return [phoneme]

def expand_phoneme_sequence(phoneme_sequence: List[str]) -> List[str]:
    expanded = []
    for phoneme in phoneme_sequence:
        expanded.extend(expand_diphthong(phoneme))
    return expanded

def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_points_from_json(path: str) -> List[Point]:
    data = read_json(path)
    return [(float(p['x']), float(p['y'])) for p in data]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def clamp_points_to_image(points: List[Point], img_shape) -> List[Point]:
    h, w = img_shape[:2]
    clamped = []
    for x, y in points:
        x = max(1, min(w-2, x))
        y = max(1, min(h-2, y))
        clamped.append((x, y))
    return clamped

def add_boundary_points(points: List[Point], img_shape, is_tongue=False) -> List[Point]:
    h, w = img_shape[:2]
    
    enhanced_points = points.copy()
    
    corner_points = [
        (0, 0),
        (w-1, 0),
        (w-1, h-1),
        (0, h-1)
    ]
    
    edge_density = 50 if is_tongue else 20
    
    for i in range(1, edge_density):
        x = (w-1) * i / edge_density
        enhanced_points.append((x, 0))
    
    for i in range(1, edge_density):
        x = (w-1) * i / edge_density
        enhanced_points.append((x, h-1))
    
    for i in range(1, edge_density):
        y = (h-1) * i / edge_density
        enhanced_points.append((0, y))
    
    for i in range(1, edge_density):
        y = (h-1) * i / edge_density
        enhanced_points.append((w-1, y))
    
    enhanced_points.extend(corner_points)
    
    return enhanced_points

def rect_from_shape(img):
    h,w = img.shape[:2]
    return (0,0,w,h)

def calculate_delaunay_triangles(rect, points: List[Point]):
    w, h = rect[2], rect[3]
    safe_points = []
    for x, y in points:
        x = max(1, min(w-2, x))
        y = max(1, min(h-2, y))
        safe_points.append((x, y))
    
    subdiv = cv2.Subdiv2D(rect)
    
    for i, p in enumerate(safe_points):
        try:
            subdiv.insert((p[0], p[1]))
        except cv2.error as e:
            print(f"Warning: Failed to insert point {i} ({p[0]}, {p[1]}): {e}")
            continue
    
    try:
        triangleList = subdiv.getTriangleList()
    except cv2.error as e:
        print(f"Error getting triangle list: {e}")
        return []
    
    pts = np.array(safe_points)
    tri_indices = []
    
    def find_index(pt):
        if len(pts) == 0:
            return None
        d = np.linalg.norm(pts - np.array(pt), axis=1)
        idx = int(np.argmin(d))
        if d[idx] < 3.0:
            return idx
        return None
    
    for t in triangleList:
        triangle_points = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        
        valid_triangle = True
        for tp in triangle_points:
            if tp[0] < 0 or tp[0] >= w or tp[1] < 0 or tp[1] >= h:
                valid_triangle = False
                break
        
        if not valid_triangle:
            continue
            
        idxs = [find_index(p) for p in triangle_points]
        if None not in idxs and len(set(idxs)) == 3:
            tri_indices.append(tuple(idxs))
    
    seen = set()
    uniq = []
    for tri in tri_indices:
        if tri not in seen:
            seen.add(tri)
            uniq.append(tri)
    return uniq

def apply_affine_transform(src, src_tri, dst_tri, size):
    src_tri = np.array(src_tri, dtype=np.float32)
    dst_tri = np.array(dst_tri, dtype=np.float32)
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img_src, img_dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] ==0 or r1[3]==0 or r2[2]==0 or r2[3]==0:
        return
    t1_rect, t2_rect, t2_rect_int = [], [], []
    for i in range(3):
        t1_rect.append(((t_src[i][0] - r1[0]),(t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]),(t_dst[i][1] - r2[1])))
        t2_rect_int.append((int(t_dst[i][0] - r2[0]), int(t_dst[i][1] - r2[1])))
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)
    img1_rect = img_src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    dst_region = img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    dst_region[:] = dst_region * (1 - mask) + img2_rect * mask
    img_dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_region

def linear_interpolate_points(p1: List[Point], p2: List[Point], t: float) -> List[Point]:
    if len(p1) != len(p2):
        raise ValueError(f"Point lists must have same length: {len(p1)} vs {len(p2)}")
    return [((1-t)*x1 + t*x2, (1-t)*y1 + t*y2) for (x1,y1),(x2,y2) in zip(p1,p2)]

def enhance_landmarks_with_boundary(pts1: List[Point], pts2: List[Point], img_shape, is_tongue=False) -> Tuple[List[Point], List[Point]]:
    pts1 = clamp_points_to_image(pts1, img_shape)
    pts2 = clamp_points_to_image(pts2, img_shape)
    
    enhanced_pts1 = add_boundary_points(pts1, img_shape, is_tongue)
    enhanced_pts2 = add_boundary_points(pts2, img_shape, is_tongue)
    
    if len(pts1) != len(pts2):
        raise ValueError(f"Landmark counts don't match: {len(pts1)} vs {len(pts2)}")
    
    num_original_landmarks = len(pts1)
    num_boundary_points = len(enhanced_pts1) - num_original_landmarks
    
    enhanced_pts2 = pts2.copy()
    enhanced_pts2.extend(enhanced_pts1[num_original_landmarks:])
    
    enhanced_pts1 = clamp_points_to_image(enhanced_pts1, img_shape)
    enhanced_pts2 = clamp_points_to_image(enhanced_pts2, img_shape)
    
    return enhanced_pts1, enhanced_pts2

def morph_two_images(img1, img2, pts1: List[Point], pts2: List[Point], t: float, is_tongue=False):
    if img1.shape != img2.shape:
        raise ValueError('Images must be same shape.')
    
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    try:
        enhanced_pts1, enhanced_pts2 = enhance_landmarks_with_boundary(pts1, pts2, img1.shape, is_tongue)
    except Exception as e:
        print(f"Error enhancing landmarks: {e}")
        return img1.astype(np.uint8)
    
    print(f"Original landmarks: {len(pts1)}, Enhanced points: {len(enhanced_pts1)}")
    
    try:
        points_mid = linear_interpolate_points(enhanced_pts1, enhanced_pts2, t)
        points_mid = clamp_points_to_image(points_mid, img1.shape)
    except Exception as e:
        print(f"Error interpolating points: {e}")
        return img1.astype(np.uint8)
    
    rect = rect_from_shape(img1)
    try:
        tri_idxs = calculate_delaunay_triangles(rect, points_mid)
    except Exception as e:
        print(f"Error calculating triangles: {e}")
        return img1.astype(np.uint8)
    
    if len(tri_idxs) == 0:
        print("Warning: No valid triangles found, returning original image")
        return img1.astype(np.uint8)
    
    print(f"Number of triangles: {len(tri_idxs)}")
    
    img1_warped = np.zeros(img1.shape, dtype=np.float32)
    img2_warped = np.zeros(img2.shape, dtype=np.float32)
    
    successful_triangles = 0
    for tri in tri_idxs:
        try:
            x, y, z = tri
            t1 = [enhanced_pts1[x], enhanced_pts1[y], enhanced_pts1[z]]
            t2 = [enhanced_pts2[x], enhanced_pts2[y], enhanced_pts2[z]]
            t_mid = [points_mid[x], points_mid[y], points_mid[z]]
            
            warp_triangle(img1, img1_warped, t1, t_mid)
            warp_triangle(img2, img2_warped, t2, t_mid)
            successful_triangles += 1
        except Exception as e:
            print(f"Warning: Failed to warp triangle {tri}: {e}")
            continue
    
    print(f"Successfully warped {successful_triangles}/{len(tri_idxs)} triangles")
    
    result = (1.0 - t) * img1_warped + t * img2_warped
    return np.uint8(np.clip(result, 0, 255))

def load_phoneme_map(mapping_path=MAPPING_JSON) -> Dict:
    return read_json(mapping_path)

def load_image_for_entry(entry: Dict, base_dir=BASE_DIR):
    img_path = os.path.join(IMAGES_DIR, entry['image'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        raise FileNotFoundError(f'Image not found: {img_path}')
    print(f"Found image: {img_path}")
    return img

def load_landmarks_for_entry(entry: Dict, base_dir=BASE_DIR):
    lm_path = os.path.join(LANDMARKS_DIR, entry['landmarks'])
    pts = read_points_from_json(lm_path)
    return pts

def assemble_entry(label_key: str, mapping: Dict) -> Dict:
    entry = mapping['phoneme_mapping'].get(label_key)
    if not entry:
        if '_반모음_' in label_key:
            parts = label_key.split('_')
            if len(parts) >= 3:
                semivowel_type = f"{parts[0]}_반모음"
                target_type = parts[2]
                try:
                    return create_semivowel_entry(semivowel_type, target_type)
                except KeyError:
                    pass
        raise KeyError(f'Label not found: {label_key}')
    entry = entry.copy()
    entry['label'] = label_key
    return entry

def generate_transition(fr_from: Dict, fr_to: Dict, frames:int=10, fps:int=10, out_root=OUTPUT_DIR):
    ensure_dir(out_root)
    
    is_tongue = '_혀' in fr_from.get('label', '') or '_tongue' in fr_from.get('label', '')
    
    hangul_to_english = {
        'ㅏ': 'a', 'ㅓ': 'eo', 'ㅗ': 'o', 'ㅜ': 'u', 'ㅡ': 'eu', 'ㅣ': 'i',
        'ㅑ': 'ya', 'ㅕ': 'yeo', 'ㅛ': 'yo', 'ㅠ': 'yu', 'ㅒ': 'yae', 'ㅖ': 'ye',
        'ㅘ': 'wa', 'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅝ': 'wo', 'ㅞ': 'we', 'ㅟ': 'wi', 'ㅢ': 'ui',
        'ㅐ': 'ae', 'ㅔ': 'e',
        'ㄱ': 'g', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄹ': 'r', 'ㅁ': 'm', 'ㅂ': 'b', 'ㅅ': 's',
        'ㅇ': 'ng', 'ㅈ': 'j', 'ㅊ': 'ch', 'ㅋ': 'k', 'ㅌ': 't', 'ㅍ': 'p', 'ㅎ': 'h',
        'ㄲ': 'gg', 'ㄸ': 'dd', 'ㅃ': 'bb', 'ㅆ': 'ss', 'ㅉ': 'jj',
        '_입': '_mouth', '_혀': '_tongue', '_반모음': '_semivowel'
    }
    
    def safe_convert_hangul(text):
        result = text
        for hangul, english in hangul_to_english.items():
            result = result.replace(hangul, english)
        return result
    
    from_safe = safe_convert_hangul(fr_from['label'])
    to_safe = safe_convert_hangul(fr_to['label'])
    
    out_dir = os.path.join(out_root, f"{from_safe}_to_{to_safe}")
    ensure_dir(out_dir)
    
    print(f"Generating {'tongue' if is_tongue else 'mouth'} transition: {fr_from['label']} → {fr_to['label']}")
    print(f"Output folder: {out_dir}")
    
    img1 = load_image_for_entry(fr_from)
    img2 = load_image_for_entry(fr_to)
    pts1 = load_landmarks_for_entry(fr_from)
    pts2 = load_landmarks_for_entry(fr_to)
    
    print(f"Image1 shape: {img1.shape}, Image2 shape: {img2.shape}")
    print(f"Points1 count: {len(pts1)}, Points2 count: {len(pts2)}")
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        print('Warning: resized target image to match source.')
    
    for i in tqdm(range(frames+1)):
        t = i / float(frames)
        try:
            morphed = morph_two_images(img1, img2, pts1, pts2, t, is_tongue)
            frame_path = os.path.join(out_dir, f'frame_{i:03d}.png')
            
            success = cv2.imwrite(frame_path, morphed)
            if success:
                if i == 0:
                    print(f"First frame saved successfully: {frame_path}")
                if not os.path.exists(frame_path):
                    print(f"Warning: Frame {i} was not saved properly")
                elif i % 5 == 0:
                    file_size = os.path.getsize(frame_path)
                    if file_size < 1000:
                        print(f"Warning: Frame {i} file size too small: {file_size} bytes")
            else:
                print(f"Error: Failed to save frame {i}")
                
        except Exception as e:
            print(f"Error generating frame {i}: {e}")
            return
    
    try:
        h, w = img1.shape[:2]
        video_path = os.path.join(out_root, f"{from_safe}_to_{to_safe}.mp4")
        
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'XVID'),
            cv2.VideoWriter_fourcc(*'MJPG'),
            cv2.VideoWriter_fourcc(*'mp4v'),
        ]
        
        vw = None
        for i, fourcc in enumerate(fourcc_options):
            try:
                if i == 0:
                    video_path = video_path.replace('.mp4', '.avi')
                vw = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                if vw.isOpened():
                    print(f"Using codec: {['XVID', 'MJPG', 'mp4v'][i]}")
                    break
                vw.release()
            except:
                continue
        
        if vw is None or not vw.isOpened():
            print("Warning: Could not create video writer, frames saved as images only")
            return
        
        frames_written = 0
        for i in range(frames+1):
            frame_path = os.path.join(out_dir, f'frame_{i:03d}.png')
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    vw.write(frame)
                    frames_written += 1
                else:
                    print(f"Could not read frame: {frame_path}")
        
        vw.release()
        print(f"Video saved: {video_path} ({frames_written} frames)")
        
    except Exception as e:
        print(f"Error creating video: {e}")
    
    print(f"Done: {frames+1} frames generated in {out_dir}")

def generate_sequence(phoneme_sequence: List[str], frames_per_transition:int=15, diphthong_frames:int=8, liaison_frames:int=20, fps:int=10, out_root=OUTPUT_DIR):
    ensure_dir(out_root)
    
    mapping = load_phoneme_map()
    
    expanded_sequence = expand_phoneme_sequence(phoneme_sequence)
    print(f"Original sequence: {phoneme_sequence}")
    print(f"Expanded sequence: {expanded_sequence}")
    
    entries = []
    missing_phonemes = []
    for phoneme in expanded_sequence:
        try:
            entry = assemble_entry(phoneme, mapping)
            entries.append(entry)
        except KeyError:
            missing_phonemes.append(phoneme)
    
    if missing_phonemes:
        print(f"Error: The following phonemes are missing from mapping:")
        for mp in missing_phonemes:
            print(f"  - {mp}")
        print(f"\nPlease add these to your phoneme_mapping.json or create the corresponding images/landmarks.")
        return
    
    print(f"Generating sequence: {' → '.join(expanded_sequence)}")
    
    hangul_to_english = {
        'ㅏ': 'a', 'ㅓ': 'eo', 'ㅗ': 'o', 'ㅜ': 'u', 'ㅡ': 'eu', 'ㅣ': 'i',
        'ㅑ': 'ya', 'ㅕ': 'yeo', 'ㅛ': 'yo', 'ㅠ': 'yu', 'ㅒ': 'yae', 'ㅖ': 'ye',
        'ㅘ': 'wa', 'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅝ': 'wo', 'ㅞ': 'we', 'ㅟ': 'wi', 'ㅢ': 'ui',
        'ㅐ': 'ae', 'ㅔ': 'e',
        'ㄱ': 'g', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄹ': 'r', 'ㅁ': 'm', 'ㅂ': 'b', 'ㅅ': 's',
        'ㅇ': 'ng', 'ㅈ': 'j', 'ㅊ': 'ch', 'ㅋ': 'k', 'ㅌ': 't', 'ㅍ': 'p', 'ㅎ': 'h',
        'ㄲ': 'gg', 'ㄸ': 'dd', 'ㅃ': 'bb', 'ㅆ': 'ss', 'ㅉ': 'jj',
        '_입': '_mouth', '_혀': '_tongue', '_반모음': '_semivowel'
    }
    
    def safe_convert_hangul(text):
        result = text
        for hangul, english in hangul_to_english.items():
            result = result.replace(hangul, english)
        return result
    
    safe_original = []
    for p in phoneme_sequence:
        safe_p = safe_convert_hangul(p)
        safe_original.append(safe_p)
    
    sequence_name = "_".join(safe_original)
    out_dir = os.path.join(out_root, f"sequence_{sequence_name}")
    ensure_dir(out_dir)
    
    all_frames = []
    total_frame_count = 0
    
    for i in range(len(entries) - 1):
        fr_from = entries[i]
        fr_to = entries[i + 1]
        
        is_silent_start = ('ㅇ_' in fr_from.get('label', '') and 
                          i == 0)
        
        if is_silent_start:
            print(f"Skipping silent start: {fr_from['label']} (무성음)")
            continue
        
        is_liaison = (fr_from.get('label', '').split('_')[0] == fr_to.get('label', '').split('_')[0] and
                     fr_from.get('label', '').split('_')[0] in ['ㄴ', 'ㄹ', 'ㅁ', 'ㅇ'])
        
        is_diphthong_transition = ('반모음' in fr_from.get('label', '') or 
                                 '반모음' in fr_to.get('label', ''))
        
        if is_liaison:
            current_frames = liaison_frames
            transition_type = "liaison"
        elif is_diphthong_transition:
            current_frames = diphthong_frames
            transition_type = "diphthong"
        else:
            current_frames = frames_per_transition
            transition_type = "normal"
        
        print(f"\nGenerating transition {i+1}/{len(entries)-1}: {fr_from['label']} → {fr_to['label']} ({current_frames} frames, {transition_type})")
        
        img1 = load_image_for_entry(fr_from)
        img2 = load_image_for_entry(fr_to)
        pts1 = load_landmarks_for_entry(fr_from)
        pts2 = load_landmarks_for_entry(fr_to)
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        is_tongue = '_혀' in fr_from.get('label', '') or '_tongue' in fr_from.get('label', '')
        
        transition_frames = []
        for j in tqdm(range(current_frames + 1), desc=f"Transition {i+1}"):
            t = j / float(current_frames)
            try:
                morphed = morph_two_images(img1, img2, pts1, pts2, t, is_tongue)
                
                if i > 0 and j == 0:
                    continue
                    
                frame_path = os.path.join(out_dir, f'frame_{total_frame_count:04d}.png')
                cv2.imwrite(frame_path, morphed)
                transition_frames.append(morphed)
                total_frame_count += 1
                
            except Exception as e:
                print(f"Error generating frame {j} in transition {i+1}: {e}")
                return
        
        all_frames.extend(transition_frames)
        
    print(f"\nTotal frames generated: {total_frame_count}")
    
    try:
        if all_frames:
            h, w = all_frames[0].shape[:2]
            video_path = os.path.join(out_root, f"sequence_{sequence_name}.mp4")
            
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'XVID'),
                cv2.VideoWriter_fourcc(*'MJPG'),
                cv2.VideoWriter_fourcc(*'mp4v'),
            ]
            
            vw = None
            for i, fourcc in enumerate(fourcc_options):
                try:
                    if i == 0:
                        video_path = video_path.replace('.mp4', '.avi')
                    vw = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                    if vw.isOpened():
                        print(f"Using codec: {['XVID', 'MJPG', 'mp4v'][i]}")
                        break
                    vw.release()
                except:
                    continue
            
            if vw is None or not vw.isOpened():
                print("Warning: Could not create video writer, frames saved as images only")
                return
            
            frames_written = 0
            for i in range(total_frame_count):
                frame_path = os.path.join(out_dir, f'frame_{i:04d}.png')
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        if frame.shape[:2] != (h, w):
                            frame = cv2.resize(frame, (w, h))
                        vw.write(frame)
                        frames_written += 1
            
            vw.release()
            
            duration = frames_written / fps
            print(f"Sequence video saved: {video_path}")
            print(f"Duration: {duration:.1f} seconds ({frames_written} frames at {fps}fps)")
            
    except Exception as e:
        print(f"Error creating sequence video: {e}")
        print("Individual frames are saved successfully in the output folder")

def main_cli():
    parser = argparse.ArgumentParser(description='Image morphing for phonemes')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    single_parser = subparsers.add_parser('single', help='Generate single transition')
    single_parser.add_argument('--from', dest='frm', type=str, required=True, help='source key (e.g. ㅏ_혀)')
    single_parser.add_argument('--to', dest='to', type=str, required=True, help='target key (e.g. ㅔ_혀)')
    single_parser.add_argument('--frames', type=int, default=30, help='number of intermediate frames')
    single_parser.add_argument('--fps', type=int, default=10, help='frames per second for output video')
    
    sequence_parser = subparsers.add_parser('sequence', help='Generate phoneme sequence')
    sequence_parser.add_argument('--phonemes', type=str, required=True, 
                                help='comma-separated phoneme sequence (e.g. ㅇ_혀,ㅏ_혀,ㄴ_혀,ㅕ_혀,ㅇ_혀)')
    sequence_parser.add_argument('--frames', type=int, default=15, help='frames per transition')
    sequence_parser.add_argument('--diphthong-frames', type=int, default=8, help='frames for diphthong transitions')
    sequence_parser.add_argument('--liaison-frames', type=int, default=20, help='frames for liaison (연음) transitions')
    sequence_parser.add_argument('--fps', type=int, default=10, help='frames per second for output video')
    
    parser.add_argument('--from', dest='frm', type=str, help='[DEPRECATED] use "single --from" instead')
    parser.add_argument('--to', dest='to', type=str, help='[DEPRECATED] use "single --to" instead')
    parser.add_argument('--frames', type=int, default=30, help='[DEPRECATED] use subcommands instead')
    parser.add_argument('--fps', type=int, default=10, help='[DEPRECATED] use subcommands instead')

    args = parser.parse_args()
    mapping = load_phoneme_map()

    if args.command == 'single':
        fr_entry = assemble_entry(args.frm, mapping)
        to_entry = assemble_entry(args.to, mapping)
        generate_transition(fr_entry, to_entry, frames=args.frames, fps=args.fps)
        
    elif args.command == 'sequence':
        phoneme_list = [p.strip() for p in args.phonemes.split(',')]
        diphthong_frames = getattr(args, 'diphthong_frames', 8)
        liaison_frames = getattr(args, 'liaison_frames', 20)
        generate_sequence(phoneme_list, frames_per_transition=args.frames, 
                         diphthong_frames=diphthong_frames, liaison_frames=liaison_frames, fps=args.fps)
        
    else:
        if args.frm and args.to:
            print("Warning: Using deprecated syntax. Use 'python pipeline.py single --from X --to Y' instead")
            fr_entry = assemble_entry(args.frm, mapping)
            to_entry = assemble_entry(args.to, mapping)
            generate_transition(fr_entry, to_entry, frames=args.frames, fps=args.fps)
        else:
            parser.print_help()

if __name__ == '__main__':
    main_cli()