import json
import csv
from pathlib import Path
from collections import Counter

def load_phoneme_mapping():
    try:
        with open('phoneme_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("phoneme_mapping.json 파일을 찾을 수 없습니다.")
        return None

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
        "glottal": ["ㅎ"],
        "front_vowels": ["ㅣ", "ㅔ", "ㅐ"],
        "central_vowels": ["ㅏ", "ㅓ", "ㅡ"],
        "back_vowels": ["ㅗ", "ㅜ"],
        "complex_vowels": ["ㅟ", "ㅚ"]
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

def calculate_anatomical_distance(phoneme1, phoneme2):
    position_scores = {
        "ㅣ": {"tongue_x": 8, "tongue_y": 8, "mouth_opening": 3},
        "ㅔ": {"tongue_x": 7, "tongue_y": 6, "mouth_opening": 4},
        "ㅐ": {"tongue_x": 7, "tongue_y": 5, "mouth_opening": 5},
        "ㅏ": {"tongue_x": 5, "tongue_y": 2, "mouth_opening": 8},
        "ㅓ": {"tongue_x": 4, "tongue_y": 4, "mouth_opening": 6},
        "ㅡ": {"tongue_x": 5, "tongue_y": 6, "mouth_opening": 2},
        "ㅗ": {"tongue_x": 3, "tongue_y": 7, "mouth_opening": 4},
        "ㅜ": {"tongue_x": 3, "tongue_y": 6, "mouth_opening": 3},
        "ㅟ": {"tongue_x": 6, "tongue_y": 8, "mouth_opening": 2},
        "ㅚ": {"tongue_x": 6, "tongue_y": 7, "mouth_opening": 3},
        "ㅂ": {"tongue_x": 5, "tongue_y": 5, "mouth_opening": 0},
        "ㅍ": {"tongue_x": 5, "tongue_y": 5, "mouth_opening": 0},
        "ㅁ": {"tongue_x": 5, "tongue_y": 5, "mouth_opening": 0},
        "ㄷ": {"tongue_x": 6, "tongue_y": 8, "mouth_opening": 1},
        "ㅌ": {"tongue_x": 6, "tongue_y": 8, "mouth_opening": 1},
        "ㄴ": {"tongue_x": 6, "tongue_y": 8, "mouth_opening": 1},
        "ㄱ": {"tongue_x": 3, "tongue_y": 9, "mouth_opening": 1},
        "ㅋ": {"tongue_x": 3, "tongue_y": 9, "mouth_opening": 1},
        "ㅇ": {"tongue_x": 3, "tongue_y": 9, "mouth_opening": 1},
        "ㅅ": {"tongue_x": 6, "tongue_y": 7, "mouth_opening": 2},
        "ㅈ": {"tongue_x": 7, "tongue_y": 8, "mouth_opening": 1},
        "ㅊ": {"tongue_x": 7, "tongue_y": 8, "mouth_opening": 1},
        "ㄹ": {"tongue_x": 6, "tongue_y": 7, "mouth_opening": 2},
        "ㅎ": {"tongue_x": 4, "tongue_y": 5, "mouth_opening": 3}
    }
    
    p1 = phoneme1.split('_')[0]
    p2 = phoneme2.split('_')[0]
    
    if p1 not in position_scores or p2 not in position_scores:
        return 5.0
    
    pos1 = position_scores[p1]
    pos2 = position_scores[p2]
    
    distance = ((pos1["tongue_x"] - pos2["tongue_x"])**2 + 
                (pos1["tongue_y"] - pos2["tongue_y"])**2 + 
                (pos1["mouth_opening"] - pos2["mouth_opening"])**2)**0.5
    
    return min(10.0, distance)

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

def check_anatomy_consistency(phoneme1, phoneme2):
    type1 = get_phoneme_type(phoneme1)
    type2 = get_phoneme_type(phoneme2)
    
    if type1 == type2:
        return "high"
    elif (type1 == "vowel" and type2 == "consonant") or (type1 == "consonant" and type2 == "vowel"):
        return "medium"
    else:
        return "low"

def create_transition_table(phoneme_mapping_data):
    transition_frames, articulation_groups = create_transition_rules()
    phonemes = list(phoneme_mapping_data["phoneme_mapping"].keys())
    
    transition_table = []
    
    for i, phoneme1 in enumerate(phonemes):
        for j, phoneme2 in enumerate(phonemes):
            if i != j:
                transition_type = determine_transition_type(phoneme1, phoneme2, articulation_groups)
                frame_info = transition_frames[transition_type]
                anatomical_distance = calculate_anatomical_distance(phoneme1, phoneme2)
                
                base_frames = (frame_info["min"] + frame_info["max"]) // 2
                actual_frames = max(frame_info["min"], 
                                  min(frame_info["max"], 
                                      base_frames + int(anatomical_distance * 2)))
                
                transition_entry = {
                    "from_phoneme": phoneme1,
                    "to_phoneme": phoneme2,
                    "from_image": phoneme_mapping_data["phoneme_mapping"][phoneme1]["image"],
                    "to_image": phoneme_mapping_data["phoneme_mapping"][phoneme2]["image"],
                    "from_landmarks": phoneme_mapping_data["phoneme_mapping"][phoneme1]["landmarks"],
                    "to_landmarks": phoneme_mapping_data["phoneme_mapping"][phoneme2]["landmarks"],
                    "transition_type": transition_type,
                    "min_frames": frame_info["min"],
                    "max_frames": frame_info["max"],
                    "recommended_frames": actual_frames,
                    "anatomical_distance": round(anatomical_distance, 2),
                    "morphing_difficulty": get_morphing_difficulty(transition_type),
                    "anatomy_consistency": check_anatomy_consistency(phoneme1, phoneme2)
                }
                
                transition_table.append(transition_entry)
    
    return transition_table

def create_korean_text_sequence_examples():
    examples = [
        {
            "text": "안녕",
            "sequence": ["ㅇ_입", "ㅏ_입", "ㄴ_입", "ㄴ_입", "ㅕ_입", "ㅇ_입"]
        },
        {
            "text": "사랑",
            "sequence": ["ㅅ_입", "ㅏ_입", "ㄹ_입", "ㅏ_입", "ㅇ_입"]
        },
        {
            "text": "학교",
            "sequence": ["ㅎ_입", "ㅏ_입", "ㄱ_입", "ㄱ_입", "ㅛ_입"]
        }
    ]
    return examples

def generate_transition_statistics(transition_table):
    stats = {
        "total_transitions": len(transition_table),
        "transition_type_counts": dict(Counter(t["transition_type"] for t in transition_table)),
        "difficulty_distribution": dict(Counter(t["morphing_difficulty"] for t in transition_table)),
        "average_frames_by_type": {},
        "max_anatomical_distance": max(t["anatomical_distance"] for t in transition_table),
        "min_anatomical_distance": min(t["anatomical_distance"] for t in transition_table)
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
    
    transition_data = {
        "project_name": "Korean Pronunciation Frame Generation",
        "current_step": "음소 전환 시퀀스 테이블",
        "total_transitions": len(transition_table),
        "transition_table": transition_table,
        "text_sequence_examples": examples
    }
    
    with open(output_path / "phoneme_transition_table.json", 'w', encoding='utf-8') as f:
        json.dump(transition_data, f, ensure_ascii=False, indent=2)
    
    csv_file = output_path / "phoneme_transition_table.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        if transition_table:
            writer = csv.DictWriter(f, fieldnames=transition_table[0].keys())
            writer.writeheader()
            writer.writerows(transition_table)
    
    stats = generate_transition_statistics(transition_table)
    with open(output_path / "transition_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"전환 테이블 저장 완료: {len(transition_table)}개 전환")
    print(f"파일: {output_path}")

def main():
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