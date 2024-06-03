import os


def get_the_number_of_files(root_dir):
    number = 0  # 내부에서 리스트 초기화
    for label_dir in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_dir)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    number += 1
    return number