import os


def get_image_filenames(directory: str, recursive: bool = False) -> list[str]:
    image_filenames: list[str] = []

    if recursive:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_filenames.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_filenames.append(os.path.join(directory, filename))

    return image_filenames

