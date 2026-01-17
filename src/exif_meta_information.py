from PIL import Image
from PIL.ExifTags import TAGS

def extract_exif(image_path):
    img = Image.open(image_path)
    
    exif_data = {}
    info = img._getexif()
    if info is not None:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            exif_data[decoded] = value

    return exif_data
