import cv2 # Handles reading, resizing and color-converting images.
import os  # accessing folders, browsing them and builing file paths, regardless of the OS.

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 224  # '224' choosen, due to it being the standard input size for pretained CNNs like ResNet/MobileNet.
                # Creating a variable here helps recognize it much easily, rather than scattering '224' randomly across
                # the entire codebase.

classes = os.listdir(RAW_DIR) # This attribut of 'os' module provides us with the list of everything in that specific
                              # directory. In our case ['eczema', 'psoriasis', 'fungal_infections'](folder names).

for class_name in classes:
    class_input_dir = os.path.join(RAW_DIR, class_name) # This builds the path to a class's raw folder. Ex: "data/raw/eczema".
                                                        # Much easier than manually managing the correct address across various OS.

    class_output_dir = os.path.join(PROCESSED_DIR, class_name) # Builds the path to a class's processed folder, Ex: "data/processed/eczema."

    os.makedirs(class_output_dir, exist_ok = True) # Since class_output_dir's folder doesn't yet exist, this creates it, exist_ok = True,
                                                   # makes sure that this statement doesn't throw an error incase the folder is already
                                                   # present(if the script is run more than once, or the user created it manually, etc...).

    image_files = os.listdir(class_input_dir) # Provides the list of the image files in the specific folder, Ex: ['0_0.jpg', '0_1.jpg', '0_2.jpg',....]
    
    for filename in image_files:
        input_path = os.path.join(class_input_dir, filename)      # The input_path and output_path are the actual addresses of the image file
        output_path = os.path.join(class_output_dir, filename)    # itself, Ex: "data/raw/eczema/0_0.jpg" to read from, and "data/processed/eczema/0_0.jpg"
                                                                  # to write to.

        image = cv2.imread(input_path) # This function from the 'cv2' module reads the image from the disk into memory as a NumPy array.
                                       # This is also the point where the image becomes actual data that can be maipulated.

        if image is None: # This if block exists purely to catch corrupt images that return 'None' if not read properly.
            print(f"Skipping unreadable file: {input_path}")
            continue # safeguard the code from crashing incase there is a problem reading an image.

        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) # This forces image of nxn pixels exactly into 224x224 pixels, by stretching or squashing it.
        cv2.imwrite(output_path, image_resized) # Writes the processed image to a address on the disk, Ex: "data/processed/eczema/0_0.jpg"

    print(f"Done: {class_name}")
