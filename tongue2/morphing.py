'''main에서 생성된 landmark 파일 기반 >> morphing & transition rules -> phoneme_mapping.csv '''
import json
import csv
import numpy as np
from pathlib import Path
from collections import Counter

def load_phoneme_mapping():
    # 실제 경로 추가
    possible_paths = [
        'phoneme_mapping.json',  # 현재 폴더
        r'C:\Users\user\PycharmProjects\betterpronunciation2\tongue2\phoneme_mapping.json',  # 실제 경로
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    
    print("phoneme_mapping.json 파일을 찾을 수 없습니다.")
    print(f"시도한 경로들: {possible_paths}")
    return None

def load_landmark_data(landmarks_file):
    # results 폴더에서 랜드마크 파일 찾기
    possible_paths = [
        landmarks_file,  # 현재 폴더
        f"results/{landmarks_file}",  # results 하위 폴더
        f"./results/{landmarks_file}",  # 명시적 results 폴더
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    
    print(f"랜드마크 파일을 찾을 수 없습니다: {landmarks_file}")
    print(f"시도한 경로들: {possible_paths}")
    return None

def extract_anatomical_features(landmarks_data, anatomy_type):
    if not landmarks_data:
        return None
        
    features = {}
    
    if anatomy_type == "mouth":
        left_corner = next(p for p in landmarks_data if p["name"] == "Left_Corner_1")
        right_corner = next(p for p in landmarks_data if p["name"] == "Right_Corner_1")
        
        upper_points = [p for p in landmarks_data if "Upper" in p["name"]]
        lower_points = [p for p in landmarks_data if "Lower" in p["name"]]
        
        mouth_width = abs(right_corner["x"] - left_corner["x"])
        upper_center_y = np.mean([p["y"] for p in upper_points[3:7]])
        lower_center_y = np.mean([p["y"] for p in lower_points[3:7]])
        mouth_opening = abs(lower_center_y - upper_center_y)
        
        features = {
            "mouth_width": mouth_width,
            "mouth_opening": mouth_opening,
            "lip_protrusion": mouth_width / 150.0
        }
        
    elif anatomy_type == "tongue":
        tongue_tip = next(p for p in landmarks_data if p["name"] == "Tongue_Tip")
        superior_points = [p for p in landmarks_data if "Superior" in p["name"]]
        posterior_points = [p for p in landmarks_data if "Posterior" in p["name"]]
        inferior_points = [p for p in landmarks_data if "Inferior" in p["name"]]
        
        superior_y = np.mean([p["y"] for p in superior_points[:5]])
        inferior_y = np.mean([p["y"] for p in inferior_points[:3]])
        tongue_height = abs(inferior_y - superior_y)
        
        posterior_x = np.mean([p["x"] for p in posterior_points])
        tongue_length = abs(posterior_x - tongue_tip["x"])
        
        features = {
            "tongue_tip_x": tongue_tip["x"],
            "tongue_tip_y": tongue_tip["y"],
            "tongue_height": tongue_height,
            "tongue_length": tongue_length,
            "tongue_front_back": tongue_tip["x"] / 300.0,
            "tongue_high_low": (150 - tongue_tip["y"]) / 150.0
        }
    
    return features

def calculate_same_anatomy_distance(phoneme1_key, phoneme2_key, phoneme_mapping):
    """같은 해부구조끼리 모든 랜드마크 점들의 유클리드 거리 계산"""
    
    phoneme1_data = phoneme_mapping.get(phoneme1_key, {})
    phoneme2_data = phoneme_mapping.get(phoneme2_key, {})
    
    landmarks1_file = phoneme1_data.get("landmarks")
    landmarks2_file = phoneme2_data.get("landmarks")
    anatomy_type = phoneme1_data.get("anatomy")
    
    landmarks1 = load_landmark_data(landmarks1_file) if landmarks1_file else None
    landmarks2 = load_landmark_data(landmarks2_file) if landmarks2_file else None
    
    if not landmarks1 or not landmarks2:
        return {"average_distance": 5.0, "point_distances": [], "total_distance": 0.0}
    
    # 모든 랜드마크 점들의 유클리드 거리 계산
    point_distances = []
    total_distance = 0.0
    point_count = min(len(landmarks1), len(landmarks2))
    
    for i in range(point_count):
        point1 = landmarks1[i]
        point2 = landmarks2[i]
        
        # 각 점의 유클리드 거리
        point_distance = np.sqrt(
            (point1["x"] - point2["x"])**2 + 
            (point1["y"] - point2["y"])**2
        )
        
        point_distances.append({
            "point_name": point1.get("name", f"Point_{i}"),
            "from_coords": {"x": point1["x"], "y": point1["y"]},
            "to_coords": {"x": point2["x"], "y": point2["y"]},
            "distance": round(point_distance, 2)
        })
        
        total_distance += point_distance
    
    # 평균 거리
    average_distance = total_distance / point_count if point_count > 0 else 5.0
    
    # 샘플 디버깅
    if (phoneme1_key == "ㅏ_입" and phoneme2_key == "ㅓ_입") or \
       (phoneme1_key == "ㅏ_혀" and phoneme2_key == "ㅓ_혀"):
        print(f"Sample: {phoneme1_key} -> {phoneme2_key} = {average_distance:.2f} (total: {total_distance:.2f}, points: {point_count})")
    
    return {
        "average_distance": average_distance,
        "point_distances": point_distances,
        "total_distance": total_distance,
        "point_count": point_count
    }

def create_transition_rules():
    transition_frames = {
        "vowel_to_vowel": {"min": 10, "max": 15},
        "consonant_to_consonant": {"min": 15, "max": 20},
        "consonant_to_vowel": {"min": 8, "max": 12},
        "vowel_to_consonant": {"min": 12, "max": 18},
        "same_position": {"min": 3, "max": 5},
        "rest_transition": {"min": 5, "max": 8}
    }
    
    articulation_groups = {
        "bilabial": ["ㅂ", "ㅃ", "ㅍ", "ㅁ"],
        "alveolar": ["ㄷ", "ㄸ", "ㅌ", "ㄴ", "ㅅ", "ㅆ"],
        "palatal": ["ㅈ", "ㅉ", "ㅊ"],
        "velar": ["ㄱ", "ㄲ", "ㅋ", "ㅇ"],
        "liquid": ["ㄹ"],
        "glottal": ["ㅎ"]
    }
    
    return transition_frames, articulation_groups

def get_phoneme_type(phoneme_key):
    if "휴지" in phoneme_key:
        return "rest"
    
    consonants = ["ㄱ", "ㄲ", "ㅋ", "ㄴ", "ㄷ", "ㄸ", "ㅌ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅍ", 
                  "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅎ"]
    
    actual_phoneme = phoneme_key.split('_')[0]
    return "consonant" if actual_phoneme in consonants else "vowel"

def is_same_articulation_position(phoneme1, phoneme2, articulation_groups):
    p1 = phoneme1.split('_')[0]
    p2 = phoneme2.split('_')[0]
    
    for group_name, phonemes in articulation_groups.items():
        if p1 in phonemes and p2 in phonemes:
            return True
    return False

def determine_transition_type(phoneme1, phoneme2, articulation_groups):
    if "휴지" in phoneme1 or "휴지" in phoneme2:
        return "rest_transition"
    
    type1 = get_phoneme_type(phoneme1)
    type2 = get_phoneme_type(phoneme2)
    
    if is_same_articulation_position(phoneme1, phoneme2, articulation_groups):
        return "same_position"
    
    if type1 == "vowel" and type2 == "vowel":
        return "vowel_to_vowel"
    elif type1 == "consonant" and type2 == "consonant":
        return "consonant_to_consonant"
    elif type1 == "consonant" and type2 == "vowel":
        return "consonant_to_vowel"
    elif type1 == "vowel" and type2 == "consonant":
        return "vowel_to_consonant"
    else:
        return "vowel_to_vowel"

def get_morphing_difficulty(transition_type):
    difficulty_map = {
        "vowel_to_vowel": "easy",
        "consonant_to_vowel": "medium", 
        "vowel_to_consonant": "medium",
        "consonant_to_consonant": "hard",
        "same_position": "easy",
        "rest_transition": "easy"
    }
    return difficulty_map.get(transition_type, "medium")

def create_transition_table(phoneme_mapping_data):
    transition_frames, articulation_groups = create_transition_rules()
    phonemes = list(phoneme_mapping_data["phoneme_mapping"].keys())
    
    transition_table = []
    
    print(f"실제 랜드마크 기반 전환 테이블 생성 중... ({len(phonemes)}개 음소)")
    
    for i, phoneme1 in enumerate(phonemes):
        for j, phoneme2 in enumerate(phonemes):
            if i != j:
                # 같은 해부구조끼리만 전환 허용
                anatomy1 = phoneme_mapping_data["phoneme_mapping"][phoneme1]["anatomy"]
                anatomy2 = phoneme_mapping_data["phoneme_mapping"][phoneme2]["anatomy"]
                
                if anatomy1 != anatomy2:
                    continue  # 다른 해부구조끼리는 건너뛰기
                
                transition_type = determine_transition_type(phoneme1, phoneme2, articulation_groups)
                frame_info = transition_frames[transition_type]
                
                real_distance_data = calculate_same_anatomy_distance(
                    phoneme1, phoneme2, phoneme_mapping_data["phoneme_mapping"]
                )
                
                if isinstance(real_distance_data, dict):
                    real_distance = real_distance_data["average_distance"]
                    point_distances = real_distance_data["point_distances"]
                    total_distance = real_distance_data["total_distance"]
                    point_count = real_distance_data["point_count"]
                else:
                    real_distance = real_distance_data
                    point_distances = []
                    total_distance = 0.0
                    point_count = 0
                
                base_frames = (frame_info["min"] + frame_info["max"]) // 2
                distance_factor = min(real_distance / 20.0, 1.0)
                actual_frames = max(frame_info["min"], 
                                   min(frame_info["max"], 
                                       base_frames + int(distance_factor * 5)))
                
                transition_entry = {
                    "from_phoneme": phoneme1,
                    "to_phoneme": phoneme2,
                    "from_image": phoneme_mapping_data["phoneme_mapping"][phoneme1]["image"],
                    "to_image": phoneme_mapping_data["phoneme_mapping"][phoneme2]["image"],
                    "from_landmarks": phoneme_mapping_data["phoneme_mapping"][phoneme1]["landmarks"],
                    "to_landmarks": phoneme_mapping_data["phoneme_mapping"][phoneme2]["landmarks"],
                    "anatomy": anatomy1,
                    "transition_type": transition_type,
                    "min_frames": frame_info["min"],
                    "max_frames": frame_info["max"],
                    "recommended_frames": actual_frames,
                    "real_anatomical_distance": round(real_distance, 2),
                    "total_distance": round(total_distance, 2),
                    "point_count": point_count,
                    "point_distances": point_distances,
                    "morphing_difficulty": get_morphing_difficulty(transition_type)
                }
                
                transition_table.append(transition_entry)
    
    return transition_table

def create_korean_text_sequence_examples():
    examples = [
        {
            "text": "안녕",
            "simple_sequence": ["ㅏ", "ㄴ", "ㅕ"],
            "full_sequence": ["ㅇ_입", "ㅏ_입", "ㄴ_입", "ㄴ_입", "ㅕ_입", "ㅇ_입"],
            "diphthong_morphing": {
                "ㅕ": ["ㅣ", "j-glide", "ㅓ"],
                "frames": [3, 4, 7]
            }
        },
        {
            "text": "안녕하세요",
            "simple_sequence": ["ㅏ", "ㄴ", "ㅕ", "ㅎ", "ㅏ", "ㅅ", "ㅔ", "ㅛ"],
            "diphthong_morphing": {
                "ㅕ": ["ㅣ", "j-glide", "ㅓ"],
                "ㅛ": ["ㅣ", "j-glide", "ㅗ"]
            }
        }
    ]
    return examples

def generate_transition_statistics(transition_table):
    stats = {
        "total_transitions": len(transition_table),
        "transition_type_counts": dict(Counter(t["transition_type"] for t in transition_table)),
        "difficulty_distribution": dict(Counter(t["morphing_difficulty"] for t in transition_table)),
        "average_frames_by_type": {},
        "max_real_distance": max(t["real_anatomical_distance"] for t in transition_table),
        "min_real_distance": min(t["real_anatomical_distance"] for t in transition_table),
        "avg_real_distance": round(np.mean([t["real_anatomical_distance"] for t in transition_table]), 2)
    }
    
    type_frames = {}
    for transition in transition_table:
        t_type = transition["transition_type"]
        if t_type not in type_frames:
            type_frames[t_type] = []
        type_frames[t_type].append(transition["recommended_frames"])
    
    for t_type, frames in type_frames.items():
        stats["average_frames_by_type"][t_type] = round(sum(frames) / len(frames), 1)
    
    return stats

def save_transition_table(transition_table, examples, output_dir="transition_data"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 해부구조별로 분리
    mouth_transitions = [t for t in transition_table if t.get("anatomy") == "mouth"]
    tongue_transitions = [t for t in transition_table if t.get("anatomy") == "tongue"]
    
    # 전체 데이터
    transition_data = {
        "project_name": "Korean Pronunciation Frame Generation",
        "current_step": "실제 랜드마크 기반 음소 전환 시퀀스 테이블",
        "total_transitions": len(transition_table),
        "mouth_transitions": len(mouth_transitions),
        "tongue_transitions": len(tongue_transitions),
        "distance_calculation_method": "all_landmarks_euclidean_distance",
        "transition_table": transition_table,
        "text_sequence_examples": examples
    }
    
    # 1. 전체 JSON 저장
    with open(output_path / "phoneme_transition_table_all.json", 'w', encoding='utf-8') as f:
        json.dump(transition_data, f, ensure_ascii=False, indent=2)
    
    # 2. 입 전용 JSON 저장
    mouth_data = {**transition_data, "transition_table": mouth_transitions}
    with open(output_path / "phoneme_transition_table_mouth.json", 'w', encoding='utf-8') as f:
        json.dump(mouth_data, f, ensure_ascii=False, indent=2)
    
    # 3. 혀 전용 JSON 저장
    tongue_data = {**transition_data, "transition_table": tongue_transitions}
    with open(output_path / "phoneme_transition_table_tongue.json", 'w', encoding='utf-8') as f:
        json.dump(tongue_data, f, ensure_ascii=False, indent=2)
    
    # CSV 저장 함수
    def save_csv_with_points(data, filename, max_points):
        csv_file = output_path / filename
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            if data:
                basic_fields = ['from_phoneme', 'to_phoneme', 'from_image', 'to_image', 
                               'from_landmarks', 'to_landmarks', 'anatomy', 'transition_type', 
                               'min_frames', 'max_frames', 'recommended_frames', 
                               'real_anatomical_distance', 'total_distance', 'point_count', 'morphing_difficulty']
                
                # 각 점별 거리 필드 추가
                point_fields = [f"point_{i+1}_name" for i in range(max_points)] + \
                              [f"point_{i+1}_distance" for i in range(max_points)]
                
                all_fields = basic_fields + point_fields
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writeheader()
                
                for transition in data:
                    row = {field: transition.get(field, "") for field in basic_fields}
                    
                    # 각 점별 데이터 추가
                    point_distances = transition.get("point_distances", [])
                    for i in range(max_points):
                        if i < len(point_distances):
                            row[f"point_{i+1}_name"] = point_distances[i].get("point_name", "")
                            row[f"point_{i+1}_distance"] = point_distances[i].get("distance", "")
                        else:
                            row[f"point_{i+1}_name"] = ""
                            row[f"point_{i+1}_distance"] = ""
                    
                    writer.writerow(row)
    
    # 4. CSV 파일들 저장
    save_csv_with_points(transition_table, "phoneme_transition_table_all.csv", 22)  # 최대 22개 점
    save_csv_with_points(mouth_transitions, "phoneme_transition_table_mouth.csv", 22)  # 입: 22개 점
    save_csv_with_points(tongue_transitions, "phoneme_transition_table_tongue.csv", 21)  # 혀: 21개 점
    
    # 5. 상세 분석용 CSV들 저장
    def save_detail_csv(data, filename):
        detail_csv = output_path / filename
        with open(detail_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['from_phoneme', 'to_phoneme', 'anatomy', 'point_name', 'from_x', 'from_y', 'to_x', 'to_y', 'distance'])
            
            for transition in data:
                for point_data in transition.get("point_distances", []):
                    writer.writerow([
                        transition["from_phoneme"],
                        transition["to_phoneme"], 
                        transition["anatomy"],
                        point_data.get("point_name", ""),
                        point_data.get("from_coords", {}).get("x", ""),
                        point_data.get("from_coords", {}).get("y", ""),
                        point_data.get("to_coords", {}).get("x", ""),
                        point_data.get("to_coords", {}).get("y", ""),
                        point_data.get("distance", "")
                    ])
    
    save_detail_csv(transition_table, "point_distances_detail_all.csv")
    save_detail_csv(mouth_transitions, "point_distances_detail_mouth.csv")
    save_detail_csv(tongue_transitions, "point_distances_detail_tongue.csv")
    
    # 통계 저장
    stats = generate_transition_statistics(transition_table)
    with open(output_path / "transition_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"실제 랜드마크 기반 전환 테이블 저장 완료:")
    print(f"  - 총 전환: {len(transition_table)}개")
    print(f"  - 입 전환: {len(mouth_transitions)}개")
    print(f"  - 혀 전환: {len(tongue_transitions)}개")
    print(f"파일 위치: {output_path}")
    print(f"생성된 파일들:")
    print(f"  JSON: _all.json, _mouth.json, _tongue.json")
    print(f"  CSV: _all.csv, _mouth.csv, _tongue.csv")
    print(f"  Detail: _detail_all.csv, _detail_mouth.csv, _detail_tongue.csv")
    print(f"평균 실제 거리: {stats['avg_real_distance']}")



def main():
    # 작업 디렉토리를 tongue 폴더로 변경 (phoneme_mapping.json 위치)
    import os
    base_path = r"C:\Users\user\PycharmProjects\betterpronunciation2\tongue2"
    if os.path.exists(base_path):
        os.chdir(base_path)
        print(f"작업 디렉토리 변경: {base_path}")
    
    phoneme_data = load_phoneme_mapping()
    if not phoneme_data:
        return
    
    print(f"음소 매핑 데이터 로드: {len(phoneme_data['phoneme_mapping'])}개")
    
    transition_table = create_transition_table(phoneme_data)
    examples = create_korean_text_sequence_examples()
    save_transition_table(transition_table, examples)
    
    print(f"생성 완료: {len(transition_table)}개 전환")

if __name__ == "__main__":
    main()