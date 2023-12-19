import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
__name__="__main__"
# Function to load images from a directory
def load_images_from_directory(directory):
    image_list = []
    for img_path in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, img_path))
        image_list.append(img)
    return image_list

# Main function
def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Initialize the SelfiSegmentation module
    segmentor = SelfiSegmentation()

    # Load images from the "Images" directory
    image_directory = "Images"
    img_list = load_images_from_directory(image_directory)

    # Initialize the index for cycling through images
    index_img = 0

    # Main loop
    while True:
        # Capture a frame from the camera
        success, img = cap.read()

        # Apply SelfiSegmentation to remove the background
        img_out = segmentor.removeBG(img, img_list[index_img], cutThreshold=0.9)

        # Stack the original and processed images horizontally
        img_stacked = cvzone.stackImages([img, img_out], 2, 1)

        # Display the stacked images
        cv2.imshow("Image", img_stacked)

        # Wait for a key press
        key = cv2.waitKey(1)

        # Handle key presses
        if key == ord('a') and index_img > 0:
            index_img -= 1
        elif key == ord('d') and index_img < len(img_list) - 1:
            index_img += 1
        elif key == ord('q'):
            break

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
