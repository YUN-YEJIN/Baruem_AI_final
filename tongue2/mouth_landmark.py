'''입술 모음발음 랜드마크 '''
import os
import cv2
import json

def manual_landmark_annotation_lip_vowel(image_path):
    """
    모음 발음 시 입술 모양에 이름이 부여된 랜드마크를 표시하는 함수
    (Left_Corner 1 -> Upper 1-10 -> Right_Corner 1 -> Lower 1-10)
    """
    #랜드마크의 이름과 부위별 색상을 정의
    point_definitions = (
        [{"name": f"Left_Corner_{i}", "color": (0, 255, 255)} for i in range(1, 2)] + # 노란색 (입꼬리 왼쪽)
        [{"name": f"Upper_{i}", "color": (0, 0, 255)} for i in range(1, 11)] + # 빨간색 (입 안 위쪽)
        [{"name": f"Right_Corner_{i}", "color": (0, 255, 0)} for i in range(1, 2)] + # 초록색 (입꼬리 오른쪽)
        [{"name": f"Lower_{i}", "color": (255, 0, 0)} for i in range(1, 11)]   # 파란색 (입 안 아래쪽)
    )
    total_points = len(point_definitions)
    landmarks = []
    current_point_index = 0

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지를 불러올 수 없습니다 -> '{image_path}'")
        return []

    clone = img.copy()
    cv2.namedWindow('Image')

    def click_event(event, x, y, flags, param):
        nonlocal current_point_index
        if event == cv2.EVENT_LBUTTONDOWN and current_point_index < total_points:
            point_info = point_definitions[current_point_index]
            point_name = point_info["name"]
            point_color = point_info["color"]

            # 랜드마크 정보를 딕셔너리 형태로
            landmarks.append({"name": point_name, "x": x, "y": y})
            
            # 이미지에 점, 이름 표시
            cv2.circle(img, (x, y), 3, point_color, -1)
            cv2.putText(img, str(current_point_index + 1), (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, point_color, 1)
            cv2.imshow('Image', img)
            
            print(f"'{point_name}' 랜드마크 추가됨: ({x}, {y})")
            current_point_index += 1

            # 다음 랜드마크
            if current_point_index < total_points:
                next_point_name = point_definitions[current_point_index]["name"]
                print(f"--> 다음: '{next_point_name}' 점을 클릭하세요.")
            else:
                print("\n모든 랜드마크를 찍었습니다. 'q' 키를 눌러 종료하세요.")

    cv2.setMouseCallback('Image', click_event)

    print("=== 입술 모음 발음 랜드마크 표시 시작 ===")
    print(f"이미지: '{image_path}'")
    print(f"총 {total_points}개의 랜드마크를 순서대로 클릭하세요.")
    print("\n랜드마크 순서:")
    print("1. 입꼬리 왼쪽 (노란색) 1개")
    print("2. 입 안 위쪽 (빨간색) 10개") 
    print("3. 입꼴리 오른쪽 (초록색) 1개")
    print("4. 입 안 아래쪽 (파란색) 10개")
    print("\n'q' 키를 누르면 저장 후 종료, 'r' 키를 누르면 재시작합니다.")
    print(f"--> 시작: '{point_definitions[0]['name']}' 점을 클릭하세요.")

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            img = clone.copy()
            landmarks.clear()
            current_point_index = 0
            cv2.imshow('Image', img)
            print("\n=== 재시작 ===")
            print(f"--> 시작: '{point_definitions[0]['name']}' 점을 클릭하세요.")

    cv2.destroyAllWindows()

    print("\n저장된 랜드마크 정보:")
    # 보기 좋게 출력하기 위해 json 모듈 사용
    print(json.dumps(landmarks, indent=2, ensure_ascii=False))
    return landmarks

if __name__ == "__main__":
    # 랜드마크를 찍고 싶은 입술 모음 발음 이미지 경로
    manual_annotation_image = r"C:\Users\user\PycharmProjects\betterpronunciation2\tongue2\aeiou\rest.png"

    # 결과물을 저장할 폴더
    output_folder = r"C:\Users\user\PycharmProjects\betterpronunciation2\tongue2\results"

    # --- 수동 랜드마크 표시 실행 ---
    manual_landmarks = manual_landmark_annotation_lip_vowel(manual_annotation_image)
    
    # 결과를 파일로 저장
    if manual_landmarks:
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(manual_annotation_image))[0]
        landmarks_file = os.path.join(output_folder, f"{base_filename}_mouth_landmarks.json")

        with open(landmarks_file, 'w', encoding='utf-8') as f:
            json.dump(manual_landmarks, f, indent=2, ensure_ascii=False)
        print(f"\n입술 랜드마크가 파일로 저장되었습니다: {landmarks_file}")
    
    print("\n모든 작업이 완료되었습니다.")