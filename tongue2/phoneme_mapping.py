import json
import csv
from pathlib import Path

def create_phoneme_mapping():
    phoneme_mapping = {
        "ㅏ_입": {
            "image": "A.png",
            "landmarks": "A_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅏ_혀": {
            "image": "aa.png",
            "landmarks": "aa_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅔ_입": {
            "image": "E.png",
            "landmarks": "E_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅔ_혀": {
            "image": "ee.png",
            "landmarks": "ee_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅐ_입": {
            "image": "E.png",
            "landmarks": "E_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅐ_혀": {
            "image": "ee.png",
            "landmarks": "ee_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅓ_입": {
            "image": "EO.png",
            "landmarks": "EO_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅓ_혀": {
            "image": "eo-oo.png", 
            "landmarks": "eo-oo_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅣ_입": {
            "image": "II.png",
            "landmarks": "II_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅣ_혀": {
            "image": "iiJ.png",
            "landmarks": "iiJ_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅡ_입": {
            "image": "EU.png",
            "landmarks": "EU_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅡ_혀": {
            "image": "eui-uuW.png",
            "landmarks": "eui-uuW_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅗ_입": {
            "image": "OO.png", 
            "landmarks": "OO_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅗ_혀": {
            "image": "eo-oo.png", 
            "landmarks": "eo-oo_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅜ_입": {
            "image": "UU.png",
            "landmarks": "UU_mouth_landmarks.json", 
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅜ_혀": {
            "image": "eui-uuW.png",
            "landmarks": "eui-uuW_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅟ_입": {
            "image": "WW-W.png",
            "landmarks": "WW-W_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅟ_혀": {
            "image": "iiJ.png",
            "landmarks": "iiJ_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },
        "ㅚ_입": {
            "image": "WW-W.png",
            "landmarks": "WW-W_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "vowel"
        },
        "ㅚ_혀": {
            "image": "ee.png",
            "landmarks": "ee_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "vowel"
        },

        "ㅈ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅈ_혀": {
            "image": "J.png",
            "landmarks": "J_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅉ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅉ_혀": {
            "image": "J.png",
            "landmarks": "J_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅊ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅊ_혀": {
            "image": "J.png",
            "landmarks": "J_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㄱ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㄱ_혀": {
            "image": "KNG.png",
            "landmarks": "KNG_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㄲ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㄲ_혀": {
            "image": "KNG.png",
            "landmarks": "KNG_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅋ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅋ_혀": {
            "image": "KNG.png",
            "landmarks": "KNG_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅇ_입": {
            "image": "Rest.png",
            "landmarks": "Rest_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅇ_혀": {
            "image": "KNG.png",
            "landmarks": "KNG_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㄹ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㄹ_혀": {
            "image": "L.png", 
            "landmarks": "L_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅁ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅁ_혀": {
            "image": "PM2.png",
            "landmarks": "PM2_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅂ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅂ_혀": {
            "image": "PM2.png",
            "landmarks": "PM2_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅃ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅃ_혀": {
            "image": "PM2.png",
            "landmarks": "PM2_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅍ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅍ_혀": {
            "image": "PM2.png",
            "landmarks": "PM2_manual_landmarks.json", 
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅅ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅅ_혀": {
            "image": "S.png",
            "landmarks": "S_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅆ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅆ_혀": {
            "image": "S.png",
            "landmarks": "S_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅅ_경구개_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅅ_경구개_혀": {
            "image": "SH.png",
            "landmarks": "SH_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅆ_경구개_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅆ_경구개_혀": {
            "image": "SH.png",
            "landmarks": "SH_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㄴ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㄴ_혀": {
            "image": "TN.png", 
            "landmarks": "TN_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㄷ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㄷ_혀": {
            "image": "TN.png", 
            "landmarks": "TN_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㄸ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㄸ_혀": {
            "image": "TN.png", 
            "landmarks": "TN_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅌ_입": {
            "image": "PM.png",
            "landmarks": "PM_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅌ_혀": {
            "image": "TN.png", 
            "landmarks": "TN_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "ㅎ_입": {
            "image": "Rest.png",
            "landmarks": "Rest_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "consonant"
        },
        "ㅎ_혀": {
            "image": "KNG.png",
            "landmarks": "KNG_manual_landmarks.json",
            "anatomy": "tongue",
            "type": "consonant"
        },
        "휴지_입": {
            "image": "Rest.png",
            "landmarks": "Rest_mouth_landmarks.json",
            "anatomy": "mouth",
            "type": "rest"
        }
    }
    
    mapping_table = {
        "project_name": "Korean Pronunciation Frame Generation",
        "current_step": "음소-이미지 매핑 테이블",
        "total_images": 23,
        "breakdown": {
            "mouth_images": 11,
            "tongue_images": 12,
            "total": 63
        },
        "excluded_for_morphing": {
            "j_glide_vowels": ["ㅑ", "ㅒ", "ㅕ", "ㅖ", "ㅛ", "ㅠ"],
            "w_glide_vowels": ["ㅘ", "ㅙ", "ㅝ", "ㅞ", "ㅢ"],
            "total_excluded": 22,
            "note": "모핑으로 생성할 예정"
        },
        "landmark_structure": {
            "mouth": {
                "total_points": 22,
                "structure": "Left_Corner_1 → Upper_1-10 → Right_Corner_1 → Lower_1-10",
                "generated_by": "mouth_landmark.py",
                "suffix": "_mouth_landmarks.json"
            },
            "tongue": {
                "total_points": 21, 
                "structure": "Tongue_Tip → Superior_1-10 → Posterior_1-5 → Inferior_1-5",
                "generated_by": "main.py",
                "suffix": "_manual_landmarks.json"
            }
        },
        "phoneme_mapping": phoneme_mapping
    }
    
    return mapping_table

def validate_files_exist(mapping_data, image_dir="images", landmarks_dir="landmarks"):
    print("파일 존재 여부 확인")
    
    missing_files = []
    phonemes = mapping_data["phoneme_mapping"]
    
    for phoneme_key, data in phonemes.items():
        image_path = Path(image_dir) / data["image"]
        if not image_path.exists():
            missing_files.append(f"이미지: {data['image']}")
        
        landmarks_path = Path(landmarks_dir) / data["landmarks"]
        if not landmarks_path.exists():
            missing_files.append(f"랜드마크: {data['landmarks']}")
    
    if missing_files:
        print("누락된 파일들:")
        for file in missing_files[:5]: 
            print(f"   - {file}")
        if len(missing_files) > 5:
            print(f"   ... 외 {len(missing_files)-5}개")
        return False
    else:
        print("모든 파일 존재 확인!")
        return True

def save_mapping_json(mapping_data, output_file="phoneme_mapping.json"):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"매핑 테이블 저장: {output_file}")

def save_mapping_csv(mapping_data, output_file="phoneme_mapping.csv"):
    phonemes = mapping_data["phoneme_mapping"]
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['음소키', '이미지파일', '랜드마크파일', '해부구조', '음성타입'])
        
        for key, data in phonemes.items():
            writer.writerow([
                key,
                data['image'], 
                data['landmarks'],
                data['anatomy'],
                data['type']
            ])
    
    print(f"CSV 매핑 테이블 저장: {output_file}")

def print_mapping_summary(mapping_data):
    phonemes = mapping_data["phoneme_mapping"]
    
    print(f"총 {len(phonemes)}개 음소 매핑")
    
    mouth_items = [k for k, v in phonemes.items() if v["anatomy"] == "mouth"]
    tongue_items = [k for k, v in phonemes.items() if v["anatomy"] == "tongue"] 
    
    print(f"입/입술 이미지: {len(mouth_items)}개")
    print(f"랜드마크: 22개 점 (mouth_landmark.py)")
    print(f"파일명: *_mouth_landmarks.json")
    
    print(f"혀 이미지: {len(tongue_items)}개") 
    print(f"랜드마크: 21개 점 (main.py)")
    print(f"파일명: *_manual_landmarks.json")
    
    vowels = [k for k, v in phonemes.items() if v["type"] == "vowel"]
    consonants = [k for k, v in phonemes.items() if v["type"] == "consonant"]
    rest = [k for k, v in phonemes.items() if v["type"] == "rest"]
    
    print(f"음성학적 분류:")
    print(f"  - 모음: {len(vowels)}개")
    print(f"  - 자음: {len(consonants)}개") 
    print(f"  - 휴지: {len(rest)}개")

if __name__ == "__main__":
    print("음소 매핑 테이블 생성 시작")
    
    mapping_data = create_phoneme_mapping()
    save_mapping_json(mapping_data)
    save_mapping_csv(mapping_data)
    print_mapping_summary(mapping_data)
    
    print("phoneme_mapping.json, phoneme_mapping.csv 파일이 생성되었습니다.")