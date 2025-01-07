import cv2
import os
import sys
from natsort import natsorted


def create_video_from_images(image_folder, video_name, fps=30):
    # Get all image files from the folder
    images = [img for img in os.listdir(
        image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort images by filename
    images = natsorted(images)

    # Check if there are images in the folder
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Read and write each image to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write the frame to the video

    # Release the video writer
    video.release()
    print(f"Video '{video_name}' created successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video.py task episode")
        #                       [0]   [1]    [2]
        exit(0)
    task = sys.argv[1]
    episode = sys.argv[2]
    fps = 10  # Frames per second
    image_folder = f'./saved_eval_pic/NONE/{task}/{episode}/'
    video_name = image_folder + task + ".mp4" # 'output_video.mp4'  # Desired output video name
    create_video_from_images(image_folder, video_name, fps)
    
    CBF_image_folder = f'./saved_eval_pic/CBF/{task}/{episode}/'
    CBF_video_name = CBF_image_folder + task + "-CBF.mp4" # 'output_video.mp4'  # Desired output video name
    create_video_from_images(CBF_image_folder, CBF_video_name, fps)
    
    CLF_image_folder = f'./saved_eval_pic/CLF/{task}/{episode}/'
    CLF_video_name = CLF_image_folder + task + "-CLF.mp4" # 'output_video.mp4'  # Desired output video name
    create_video_from_images(CLF_image_folder, CLF_video_name, fps)
