import json
import os

CLASS_MAPPING = {
    'bicycle': 0, 'bus': 1, 'car': 2, 'motorcycle': 3, 'other person': 4,
    'other vehicle': 5, 'pedestrian': 6, 'rider': 7, 'trailer': 8, 'train': 9, 'truck': 10
}

def convert_to_yolo_format(annotation, image_width, image_height):
    yolo_annotations = []
    
    for label in annotation['labels']:
        category = label['category']
        class_id = CLASS_MAPPING.get(category, -1)
        
        if class_id == -1:
            print(f"Warning: Unknown category '{category}' found. Skipping this label.")
            continue
        
        x1 = label['box2d']['x1']
        y1 = label['box2d']['y1']
        x2 = label['box2d']['x2']
        y2 = label['box2d']['y2']
        
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_annotations.append(yolo_line)
    
    return '\n'.join(yolo_annotations)

def process_json_file(json_file_path, output_directory, image_width, image_height):
    os.makedirs(output_directory, exist_ok=True)
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    for annotation in data:
        image_name = os.path.splitext(annotation['name'])[0]
        
        yolo_text = convert_to_yolo_format(annotation, image_width, image_height)
        
        output_file_path = os.path.join(output_directory, f"{image_name}.txt")
        with open(output_file_path, 'w') as f:
            f.write(yolo_text)
        
        print(f"Processed: {image_name}")

def main():
    base_path = '/home/dykim/lava/lava-dl/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/data/bdd100k/labels/box_track_20'
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    
    # 사용자로부터 출력 디렉토리 입력 받기
    train_output = input("Enter the output directory for train data: ")
    val_output = input("Enter the output directory for validation data: ")
    
    image_width = 1280
    image_height = 720

    # 사용자로부터 처리할 파일 수 입력 받기
    train_count = int(input("Enter the number of train JSON files to process: "))
    val_count = int(input("Enter the number of validation JSON files to process: "))

    # 트레인 데이터 처리
    train_files = sorted(os.listdir(train_dir))[:train_count]
    for file in train_files:
        json_file_path = os.path.join(train_dir, file)
        process_json_file(json_file_path, train_output, image_width, image_height)

    # 검증 데이터 처리
    val_files = sorted(os.listdir(val_dir))[:val_count]
    for file in val_files:
        json_file_path = os.path.join(val_dir, file)
        process_json_file(json_file_path, val_output, image_width, image_height)

    print("Conversion complete.")

if __name__ == "__main__":
    main()
