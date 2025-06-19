"""
Process new meatadata JSON file from original text file.
"""
import json
from tqdm import tqdm
from PIL import Image
from pathlib import Path



data_dir = "/viscam/downloads/3DAnimals/data/fauna/Fauna_dataset"
txt_suffix = "box.txt"
json_suffix = "metadata.json"

for txt in tqdm(list(Path(data_dir).rglob(f"*{txt_suffix}"))):
    json_file = str(txt).replace(txt_suffix, json_suffix)
    if Path(json_file).exists():
        continue
    print(f"Processing {txt}")
    metadata = {}
    with open(txt, 'r') as f:
        line = ' '.join(line.strip() for line in f).split()
    metadata["video_frame_id"] = int(line[0].split('_')[0])
    metadata["crop_box_xyxy"] = [
        int(float(line[1])), int(float(line[2])),
        int(float(line[1]) + float(line[3])), int(float(line[2]) + float(line[4]))
    ]
    metadata["video_frame_width"] = int(float(line[5]))
    metadata["video_frame_height"] = int(float(line[6]))
    metadata["sharpness"] = float(line[7])
    img = None
    for img_suffix in [".jpg", ".jpeg", ".png"]:
        try:
            img = Image.open(str(txt).replace(txt_suffix, f"rgb{img_suffix}"))
            break
        except FileNotFoundError:
            continue
    if img is None:
        raise FileNotFoundError(f"Image file not found for {txt}")
    width, height = img.size
    metadata["crop_height"] = int(width)
    metadata["crop_width"] = int(height)
    if len(line) == 9:
        metadata["label"] = int(float(line[8]))
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)

