from src.exif_meta_information import extract_exif
exif = extract_exif("examples\input4.jpg")  
print("EXIF Metadata:", exif)