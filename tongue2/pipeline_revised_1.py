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

BASE_DIR = r"C:\Users\user\PycharmProjects\betterpronunciation2\tongue2"
IMAGES_DIR = os.path.join(BASE_DIR, "image")
LANDMARKS_DIR = os.path.join(BASE_DIR, "results")
MAPPING_JSON = os.path.join(BASE_DIR, 'phoneme_mapping.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

SEMIVOWEL_FILES = {
    'ㅣ_반모음_혀': {'image': 'iiJ.png', 'landmarks': 'iiJ_manual_landmarks.json'},
    'ㅜ_반모음_혀': {'image': 'eui-uuW.png', 'landmarks': 'eui-uuW_manual_landmarks.json'},
    'ㅣ_반모음_입': {'image': 'JY-J.png', 'landmarks': 'JY-J_mouth_landmarks.json'},
    'ㅜ_반모음_입': {'image': 'eui-uuW.png', 'landmarks': 'eui-uuW_mouth_landmarks.json'}
}

DIPHTHONG_MAPPING = {
    'ㅕ_혀': ('ㅣ_반모음_혀', 'ㅓ_혀'), 'ㅛ_혀': ('ㅗ_혀', 'ㅅ_혀'),
    'ㅠ_혀': ('ㅣ_반모음_혀', 'ㅜ_혀'), 'ㅑ_혀': ('ㅣ_반모음_혀', 'ㅏ_혀'),
    'ㅞ_혀': ('ㅜ_반모음_혀', 'ㅔ_혀'), 'ㅟ_혀': ('ㅜ_반모음_혀', 'ㅣ_혀'),
    'ㅝ_혀': ('ㅜ_반모음_혀', 'ㅓ_혀'), 'ㅘ_혀': ('ㅗ_혀', 'ㅏ_혀'),
    'ㅢ_혀': ('ㅣ_혀'), 'ㅒ_혀': ('ㅣ_반모음_혀', 'ㅐ_혀'),
    'ㅖ_혀': ('ㅣ_반모음_혀', 'ㅔ_혀'),

    'ㅕ_입': ('ㅓ_입'), 'ㅛ_입': ('ㅗ_입'),
    'ㅠ_입': ('ㅜ_입'), 'ㅑ_입': ('ㅏ_입'),
    'ㅞ_입': ('ㅔ_입'), 'ㅟ_입': ('ㅣ_입'),
    'ㅝ_입': ('ㅓ_입'), 'ㅘ_입': ('ㅏ_입'),
    'ㅢ_입': ('ㅣ_입'), 'ㅒ_입': ('ㅐ_입'),
    'ㅖ_입': ('ㅔ_입')
}

HANGUL_TO_ENGLISH = {
    'ㅏ': 'a', 'ㅓ': 'eo', 'ㅗ': 'o', 'ㅜ': 'u', 'ㅡ': 'eu', 'ㅣ': 'i',
    'ㅑ': 'ya', 'ㅕ': 'yeo', 'ㅛ': 'yo', 'ㅠ': 'yu', 'ㅒ': 'yae', 'ㅖ': 'ye',
    'ㅘ': 'wa', 'ㅙ': 'wae', 'ㅚ': 'oe', 'ㅝ': 'wo', 'ㅞ': 'we', 'ㅟ': 'wi', 'ㅢ': 'ui',
    'ㅐ': 'ae', 'ㅔ': 'e', 'ㄱ': 'g', 'ㄴ': 'n', 'ㄷ': 'd', 'ㄹ': 'r', 'ㅁ': 'm',
    'ㅂ': 'b', 'ㅅ': 's', 'ㅇ': 'ng', 'ㅈ': 'j', 'ㅊ': 'ch', 'ㅋ': 'k', 'ㅌ': 't',
    'ㅍ': 'p', 'ㅎ': 'h', 'ㄲ': 'gg', 'ㄸ': 'dd', 'ㅃ': 'bb', 'ㅆ': 'ss', 'ㅉ': 'jj',
    '_입': '_mouth', '_혀': '_tongue', '_반모음': '_semivowel'
}

CONSONANTS = {'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'}
VOWELS = {'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅔ'}
LIAISON_CONSONANTS = {'ㄴ', 'ㄹ', 'ㅁ', 'ㅇ'}


def safe_convert_hangul(text: str) -> str:
    for hangul, english in HANGUL_TO_ENGLISH.items():
        text = text.replace(hangul, english)
    return text


def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_points_from_json(path: str) -> List[Point]:
    data = read_json(path)
    return [(float(p['x']), float(p['y'])) for p in data]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def create_semivowel_entry(semivowel_type: str, target_type: str) -> Dict:
    if semivowel_type not in SEMIVOWEL_FILES:
        raise KeyError(f"Semivowel type {semivowel_type} not found")
    files = SEMIVOWEL_FILES[semivowel_type]
    return {'label': f"{semivowel_type}_{target_type}", 'image': files['image'], 'landmarks': files['landmarks']}


def expand_diphthong(phoneme: str) -> List[str]:
    if phoneme in DIPHTHONG_MAPPING:
        mapping_value = DIPHTHONG_MAPPING[phoneme]
        target_type = phoneme.split('_')[-1]

        # 문자열 하나만 있는 경우 튜플로 변환
        if not isinstance(mapping_value, (list, tuple)):
            mapping_value = (mapping_value,)

        if len(mapping_value) == 3:
            start, semivowel_type, end = mapping_value
            return [start, f"{semivowel_type}_{target_type}", end]
        elif len(mapping_value) == 2:
            start, end = mapping_value
            return [start, end]
        elif len(mapping_value) == 1:
            return [mapping_value[0]]
        else:
            return [phoneme]
    return [phoneme]


def expand_phoneme_sequence(phoneme_sequence: List[str]) -> List[str]:
    expanded = []
    for phoneme in phoneme_sequence:
        expanded.extend(expand_diphthong(phoneme))
    return expanded


def clamp_points_to_image(points: List[Point], img_shape) -> List[Point]:
    h, w = img_shape[:2]
    return [(max(1, min(w - 2, x)), max(1, min(h - 2, y))) for x, y in points]


def add_boundary_points(points: List[Point], img_shape, is_tongue=False) -> List[Point]:
    h, w = img_shape[:2]
    enhanced_points = points.copy()
    edge_density = 50 if is_tongue else 20

    for i in range(1, edge_density):
        x = (w - 1) * i / edge_density
        enhanced_points.extend([(x, 0), (x, h - 1)])

    for i in range(1, edge_density):
        y = (h - 1) * i / edge_density
        enhanced_points.extend([(0, y), (w - 1, y)])

    enhanced_points.extend([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)])
    return enhanced_points


def calculate_delaunay_triangles(rect, points: List[Point]):
    w, h = rect[2], rect[3]
    safe_points = [(max(1, min(w - 2, x)), max(1, min(h - 2, y))) for x, y in points]

    subdiv = cv2.Subdiv2D(rect)
    for p in safe_points:
        try:
            subdiv.insert((p[0], p[1]))
        except cv2.error:
            continue

    try:
        triangleList = subdiv.getTriangleList()
    except cv2.error:
        return []

    pts = np.array(safe_points)
    tri_indices = []

    def find_index(pt):
        d = np.linalg.norm(pts - np.array(pt), axis=1)
        idx = int(np.argmin(d))
        return idx if d[idx] < 3.0 else None

    for t in triangleList:
        triangle_points = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(0 <= tp[0] < w and 0 <= tp[1] < h for tp in triangle_points):
            idxs = [find_index(p) for p in triangle_points]
            if None not in idxs and len(set(idxs)) == 3:
                tri_indices.append(tuple(idxs))

    return list(set(tri_indices))


def apply_affine_transform(src, src_tri, dst_tri, size):
    src_tri = np.array(src_tri, dtype=np.float32)
    dst_tri = np.array(dst_tri, dtype=np.float32)
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def warp_triangle(img_src, img_dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    t1_rect = [((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])) for i in range(3)]
    t2_rect = [((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])) for i in range(3)]
    t2_rect_int = [(int(t_dst[i][0] - r2[0]), int(t_dst[i][1] - r2[1])) for i in range(3)]

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img_src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, (r2[2], r2[3]))

    dst_region = img_dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    dst_region[:] = dst_region * (1 - mask) + img2_rect * mask
    img_dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst_region


def enhance_landmarks_with_boundary(pts1: List[Point], pts2: List[Point], img_shape, is_tongue=False) -> Tuple[
    List[Point], List[Point]]:
    pts1 = clamp_points_to_image(pts1, img_shape)
    pts2 = clamp_points_to_image(pts2, img_shape)

    if len(pts1) != len(pts2):
        raise ValueError(f"Landmark counts don't match: {len(pts1)} vs {len(pts2)}")

    enhanced_pts1 = add_boundary_points(pts1, img_shape, is_tongue)
    enhanced_pts2 = pts2.copy()
    enhanced_pts2.extend(enhanced_pts1[len(pts1):])

    return (clamp_points_to_image(enhanced_pts1, img_shape),
            clamp_points_to_image(enhanced_pts2, img_shape))


def smoothstep(t: float) -> float:
    """Smooth interpolation function for more natural transitions"""
    return t * t * (3 - 2 * t)


def morph_two_images(img1, img2, pts1: List[Point], pts2: List[Point], t: float, is_tongue=False):
    if img1.shape != img2.shape:
        raise ValueError('Images must be same shape.')

    img1, img2 = np.float32(img1), np.float32(img2)

    try:
        enhanced_pts1, enhanced_pts2 = enhance_landmarks_with_boundary(pts1, pts2, img1.shape, is_tongue)
        points_mid = [((1 - t) * x1 + t * x2, (1 - t) * y1 + t * y2) for (x1, y1), (x2, y2) in
                      zip(enhanced_pts1, enhanced_pts2)]
        points_mid = clamp_points_to_image(points_mid, img1.shape)

        tri_idxs = calculate_delaunay_triangles((0, 0, img1.shape[1], img1.shape[0]), points_mid)
        if not tri_idxs:
            return img1.astype(np.uint8)

        img1_warped = np.zeros(img1.shape, dtype=np.float32)
        img2_warped = np.zeros(img2.shape, dtype=np.float32)

        for tri in tri_idxs:
            try:
                x, y, z = tri
                t1 = [enhanced_pts1[x], enhanced_pts1[y], enhanced_pts1[z]]
                t2 = [enhanced_pts2[x], enhanced_pts2[y], enhanced_pts2[z]]
                t_mid = [points_mid[x], points_mid[y], points_mid[z]]
                warp_triangle(img1, img1_warped, t1, t_mid)
                warp_triangle(img2, img2_warped, t2, t_mid)
            except:
                continue

        result = (1.0 - t) * img1_warped + t * img2_warped
        return np.uint8(np.clip(result, 0, 255))

    except Exception as e:
        print(f"Morphing error: {e}")
        return img1.astype(np.uint8)


def load_phoneme_map(mapping_path=MAPPING_JSON) -> Dict:
    return read_json(mapping_path)


def load_image_for_entry(entry: Dict):
    img_path = os.path.join(IMAGES_DIR, entry['image'])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    return img


def load_landmarks_for_entry(entry: Dict):
    lm_path = os.path.join(LANDMARKS_DIR, entry['landmarks'])
    return read_points_from_json(lm_path)


def assemble_entry(label_key: str, mapping: Dict) -> Dict:
    entry = mapping['phoneme_mapping'].get(label_key)
    if not entry and '_반모음_' in label_key:
        parts = label_key.split('_')
        if len(parts) >= 3:
            try:
                return create_semivowel_entry(f"{parts[0]}_반모음_{parts[2]}", parts[2])
            except KeyError:
                pass
    if not entry:
        raise KeyError(f'Label not found: {label_key}')
    entry = entry.copy()
    entry['label'] = label_key
    return entry


def create_video(frames_dir: str, output_path: str, total_frames: int, fps: int, img_shape):
    h, w = img_shape[:2]
    fourcc_options = [cv2.VideoWriter_fourcc(*'XVID'), cv2.VideoWriter_fourcc(*'MJPG'), cv2.VideoWriter_fourcc(*'mp4v')]

    for i, fourcc in enumerate(fourcc_options):
        video_path = output_path.replace('.mp4', '.avi') if i == 0 else output_path
        vw = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if vw.isOpened():
            frames_written = 0
            for frame_idx in range(total_frames):
                frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        if frame.shape[:2] != (h, w):
                            frame = cv2.resize(frame, (w, h))
                        vw.write(frame)

                        # Add intermediate frame for smoother playback
                        next_frame_path = os.path.join(frames_dir, f'frame_{frame_idx + 1:04d}.png')
                        if os.path.exists(next_frame_path):
                            next_frame = cv2.imread(next_frame_path)
                            if next_frame is not None:
                                if next_frame.shape[:2] != (h, w):
                                    next_frame = cv2.resize(next_frame, (w, h))
                                inter_frame = cv2.addWeighted(frame, 0.5, next_frame, 0.5, 0)
                                vw.write(inter_frame)

                        frames_written += 1
            vw.release()
            if frames_written > 0:
                print(f"Video saved: {video_path} ({frames_written} frames)")
                return
    print("Warning: Could not create video, frames saved as images only")


def get_frame_count(fr_from: str, fr_to: str, i: int, total_entries: int, diphthong_frames: int,
                    liaison_frames: int, final_frames: int, frames_per_transition: int) -> int:
    from_phoneme = fr_from.split('_')[0]
    to_phoneme = fr_to.split('_')[0]

    is_final_ng = to_phoneme == 'ㅇ' and i < total_entries - 2
    is_middle_ng_transition = from_phoneme == 'ㅇ' and to_phoneme in CONSONANTS
    is_liaison = liaison_frames > 0 and from_phoneme == to_phoneme and from_phoneme in LIAISON_CONSONANTS
    is_semivowel = '반모음' in fr_from or ('반모음' in fr_to and to_phoneme in VOWELS)
    is_consonant_to_vowel = from_phoneme in CONSONANTS and to_phoneme in VOWELS and not is_liaison
    is_vowel_to_consonant = from_phoneme in VOWELS and to_phoneme in CONSONANTS and not is_liaison
    is_diphthong = '반모음' in fr_from or '반모음' in fr_to

    if is_final_ng:
        return max(3, final_frames // 3)
    elif is_middle_ng_transition:
        return max(3, diphthong_frames - 2)
    elif is_liaison:
        return max(3, liaison_frames // 2)
    elif is_semivowel:
        return max(2, diphthong_frames - 3)
    elif is_consonant_to_vowel:
        return max(3, diphthong_frames - 2)
    elif is_vowel_to_consonant:
        return max(3, diphthong_frames - 2)
    elif is_diphthong:
        return max(2, diphthong_frames - 2)
    else:
        return max(4, frames_per_transition // 2)


def get_interpolation_factor(j: int, frames: int, transition_type: str) -> float:
    """Enhanced interpolation with smoothstep for more natural transitions"""
    t_raw = j / float(frames)
    t_smooth = smoothstep(t_raw)

    if transition_type == 'semivowel':
        return smoothstep(t_smooth)
    elif transition_type == 'consonant_to_vowel':
        return smoothstep(t_smooth ** 1.2)
    elif transition_type == 'vowel_to_consonant':
        return smoothstep(t_smooth ** 1.3)
    else:
        return t_smooth


def generate_transition(fr_from: Dict, fr_to: Dict, frames: int = 10, fps: int = 10, out_root=OUTPUT_DIR):
    ensure_dir(out_root)
    is_tongue = '_혀' in fr_from.get('label', '')

    from_safe = safe_convert_hangul(fr_from['label'])
    to_safe = safe_convert_hangul(fr_to['label'])
    out_dir = os.path.join(out_root, f"{from_safe}_to_{to_safe}")
    ensure_dir(out_dir)

    img1, img2 = load_image_for_entry(fr_from), load_image_for_entry(fr_to)
    pts1, pts2 = load_landmarks_for_entry(fr_from), load_landmarks_for_entry(fr_to)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    for i in tqdm(range(frames + 1)):
        t = i / float(frames)
        morphed = morph_two_images(img1, img2, pts1, pts2, t, is_tongue)
        cv2.imwrite(os.path.join(out_dir, f'frame_{i:03d}.png'), morphed)

    create_video(out_dir, os.path.join(out_root, f"{from_safe}_to_{to_safe}.mp4"), frames + 1, fps, img1.shape)
    print(f"Done: {frames + 1} frames generated")


def generate_sequence(phoneme_sequence: List[str], frames_per_transition: int = 20, diphthong_frames: int = 10,
                      liaison_frames: int = 8, final_frames: int = 8, fps: int = 12, out_root=OUTPUT_DIR):
    ensure_dir(out_root)
    mapping = load_phoneme_map()
    expanded_sequence = expand_phoneme_sequence(phoneme_sequence)

    entries = []
    for phoneme in expanded_sequence:
        try:
            entries.append(assemble_entry(phoneme, mapping))
        except KeyError as e:
            print(f"Error: {e}")
            return

    sequence_name = "_".join(safe_convert_hangul(p) for p in phoneme_sequence)
    out_dir = os.path.join(out_root, f"sequence_{sequence_name}")
    ensure_dir(out_dir)

    total_frame_count = 0

    for i in range(len(entries) - 1):
        fr_from, fr_to = entries[i], entries[i + 1]

        if 'ㅇ_' in fr_from.get('label', '') and i == 0:
            continue

        current_frames = get_frame_count(fr_from.get('label', ''), fr_to.get('label', ''), i, len(entries),
                                         diphthong_frames, liaison_frames, final_frames, frames_per_transition)

        img1, img2 = load_image_for_entry(fr_from), load_image_for_entry(fr_to)
        pts1, pts2 = load_landmarks_for_entry(fr_from), load_landmarks_for_entry(fr_to)

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        is_tongue = '_혀' in fr_from.get('label', '')
        from_phoneme = fr_from.get('label', '').split('_')[0]
        to_phoneme = fr_to.get('label', '').split('_')[0]

        transition_type = 'normal'
        if '반모음' in fr_from.get('label', '') or ('반모음' in fr_to.get('label', '') and to_phoneme in VOWELS):
            transition_type = 'semivowel'
        elif from_phoneme in CONSONANTS and to_phoneme in VOWELS:
            transition_type = 'consonant_to_vowel'
        elif from_phoneme in VOWELS and to_phoneme in CONSONANTS:
            transition_type = 'vowel_to_consonant'

        for j in tqdm(range(current_frames + 1), desc=f"Transition {i + 1} ({fr_from['label']} → {fr_to['label']})"):
            if i > 0 and j == 0:
                continue

            t = get_interpolation_factor(j, current_frames, transition_type)
            morphed = morph_two_images(img1, img2, pts1, pts2, t, is_tongue)
            cv2.imwrite(os.path.join(out_dir, f'frame_{total_frame_count:04d}.png'), morphed)
            total_frame_count += 1

    if entries:
        last_entry = entries[-1]
        img_last = load_image_for_entry(last_entry)
        hold_frames = max(8, frames_per_transition // 2)

        for j in tqdm(range(hold_frames), desc="Holding final phoneme"):
            cv2.imwrite(os.path.join(out_dir, f'frame_{total_frame_count:04d}.png'), img_last)
            total_frame_count += 1

        create_video(out_dir, os.path.join(out_root, f"sequence_{sequence_name}.mp4"), total_frame_count, fps,
                     img1.shape)
        print(f"Total frames: {total_frame_count}, Duration: {total_frame_count / fps:.1f}s")


def main_cli():
    parser = argparse.ArgumentParser(description='Image morphing for phonemes')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    single_parser = subparsers.add_parser('single', help='Generate single transition')
    single_parser.add_argument('--from', dest='frm', type=str, required=True, help='source key')
    single_parser.add_argument('--to', dest='to', type=str, required=True, help='target key')
    single_parser.add_argument('--frames', type=int, default=30, help='number of frames')
    single_parser.add_argument('--fps', type=int, default=10, help='frames per second')

    sequence_parser = subparsers.add_parser('sequence', help='Generate phoneme sequence')
    sequence_parser.add_argument('--phonemes', type=str, required=True, help='comma-separated phoneme sequence')
    sequence_parser.add_argument('--frames', type=int, default=20, help='frames per transition (default: 20)')
    sequence_parser.add_argument('--diphthong-frames', type=int, default=10,
                                 help='frames for diphthong transitions (default: 10)')
    sequence_parser.add_argument('--liaison-frames', type=int, default=8,
                                 help='frames for liaison transitions (default: 8)')
    sequence_parser.add_argument('--final-frames', type=int, default=8,
                                 help='frames for final consonant transitions (default: 8)')
    sequence_parser.add_argument('--fps', type=int, default=12, help='frames per second (default: 12)')

    args = parser.parse_args()
    mapping = load_phoneme_map()

    if args.command == 'single':
        fr_entry = assemble_entry(args.frm, mapping)
        to_entry = assemble_entry(args.to, mapping)
        generate_transition(fr_entry, to_entry, frames=args.frames, fps=args.fps)
    elif args.command == 'sequence':
        phoneme_list = [p.strip() for p in args.phonemes.split(',')]
        generate_sequence(phoneme_list, frames_per_transition=args.frames,
                          diphthong_frames=getattr(args, 'diphthong_frames', 10),
                          liaison_frames=getattr(args, 'liaison_frames', 8),
                          final_frames=getattr(args, 'final_frames', 8), fps=args.fps)
    else:
        parser.print_help()


if __name__ == '__main__':
    main_cli()