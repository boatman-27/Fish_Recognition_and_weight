from flask import Flask, render_template, request
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import os
import math
import numpy as np
import random

app = Flask(__name__)

model_path = 'C:/Users/adham/Desktop/Work/trial/fish_recognition/fish/new_model/last.pt'
threshold = 0.5
target_width = 1280
target_height = 1280
counter = 0

# A set to store used random numbers
used_numbers = set()  

# Array to add information about the recognized Fish from the input video
recognized_fish = []

# variable to clear recognized_fish array between video inputs
measurements_done = False

# Function to add info in dictionary then in the array ^
def add_fish(class_name, Topmost_Point, Bottommost_Point, Leftmost_Point, Rightmost_Point, length, Width, Girth, Weight, Accuracy, Picture):
        fish = {
            "class_name": class_name,
            "Topmost_Point": Topmost_Point,
            "Bottommost_Point": Bottommost_Point,
            "Leftmost_Point": Leftmost_Point,
            "Rightmost_Point": Rightmost_Point, 
            "Length": length,
            "Width": Width,
            "Girth": Girth,
            "Weight": Weight,
            "Accuracy": Accuracy,
            "Picture": Picture
        }
        recognized_fish.append(fish)

def getClassName(model_path, threshold, image_path):
    name_model = YOLO(model_path)
    frame = cv2.imread(image_path)
    results = name_model(frame)[0]

    class_names = []
    accuracies = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_name = results.names[int(class_id)]
            class_names.append(class_name)
            accuracies.append(score)

    return class_names, accuracies

def getMasks(segment_model_path, image_path, target_width, target_height):
    segment_model = YOLO(segment_model_path)
    img = cv2.imread(image_path)
    results = segment_model(img)

    mask_filenames = []  # Create an array to store mask filenames

    for i, result in enumerate(results):
        masks = result.masks.data

        for j, mask in enumerate(masks):
            mask = mask.cpu().numpy() * 255
            mask = cv2.resize(mask, (target_width, target_height))

            while True:
                # Generate a random number and check if it's already used
                random_number = random.randint(1, 1000000)
                mask_filename = f"uploads/{random_number}.png"
                if random_number not in used_numbers:
                    used_numbers.add(random_number)  # Mark the number as used
                    cv2.imwrite(mask_filename, mask)
                    mask_filenames.append(mask_filename)
                    break

    return mask_filenames
    
def processVideo(segment_model_path, video_path, target_width, target_height, threshold):
    segment_model = YOLO(segment_model_path)
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    mask_filenames = []
    used_class_numbers = set()
    prev_frame = None  # Store the previous frame
    
    unique_classes = []
    accuracies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # Calculate the absolute difference between the current frame and the previous frame
            frame_diff = cv2.absdiff(prev_frame, frame)
            frame_diff_mean = np.mean(frame_diff)

            # Set a threshold for frame difference to determine if there's a significant change
            frame_diff_threshold = 10  # Adjust this threshold as needed

            if frame_diff_mean < frame_diff_threshold:
                # Skip processing this frame if it doesn't have a significant change
                continue

        results_seg = segment_model(frame)
        
        results = segment_model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                class_name = results.names[int(class_id)]
                unique_classes.append(class_name)
                accuracies.append(score)

        for i, result in enumerate(results_seg):
            for j, mask in enumerate(result.masks.data):
                mask = mask.cpu().numpy() * 255
                mask = cv2.resize(mask, (target_width, target_height))

                class_number = None
                while class_number is None or class_number in used_class_numbers:
                    class_number = np.random.randint(1, 1000)  # Generate a random number

                used_class_numbers.add(class_number)
                mask_filename = f"uploads/{class_number}_{frame_number}_{i}_{j}.png"
                cv2.imwrite(mask_filename, mask)
                mask_filenames.append(mask_filename)

        frame_number += 1
        prev_frame = frame  # Update the previous frame

    cap.release()
    return mask_filenames, unique_classes, accuracies

def getFishMeasurements(mask_filenames, class_names, accuracies):
    global measurements_done, counter
    
    if measurements_done:
        recognized_fish.clear()
        accuracies.clear()
        measurements_done = False  # Reset measurements_done
    
    for i, img in enumerate(mask_filenames):
        final_top = 0
        final_bottom = 0
        final_left = 0
        final_right = 0
        
        # Load the segmentation mask image (binary format)
        mask = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        # Create a copy of the original image for drawing lines and points
        output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store topmost, bottommost, leftmost, and rightmost points
        topmost_point = None
        bottommost_point = None
        leftmost_point = None
        rightmost_point = None

        # Create a dictionary to group points by their y-coordinate (level)
        points_by_level_y = {}

        # Create a dictionary to group points by their x-coordinate
        points_by_level_x = {}

        # Iterate through contours
        for i, contour in enumerate(contours):
            # Draw all points on the output image
            for point in contour:
                x, y = point[0]
                
                # Draw a red circle at each point
                cv2.circle(output_image, (x, y), 2, (0, 0, 255), -1)
                        
                # Update topmost and bottommost points
                if topmost_point is None or y < topmost_point[1]:
                    topmost_point = (x, y)
                if bottommost_point is None or y > bottommost_point[1]:
                    bottommost_point = (x, y)
                
                # Update leftmost and rightmost points   
                if leftmost_point is None or x < leftmost_point[0]:
                    leftmost_point = (x, y)
                if rightmost_point is None or x > rightmost_point[0]:
                    rightmost_point = (x, y)

                # Group points by their y-coordinate (level)
                if y not in points_by_level_y:
                    points_by_level_y[y] = []
                points_by_level_y[y].append((x, y))
                
                if x not in points_by_level_x:
                    points_by_level_x[x] = []
                points_by_level_x[x].append((x, y))

        # Calculate the vertical distance (length) directly from points
        vertical_distance = round(math.sqrt((bottommost_point[0] - topmost_point[0])**2 + (bottommost_point[1] - topmost_point[1])**2), 2)

        # Initialize variables to store the leftmost and rightmost points on the same level
        same_level_leftmost = None
        same_level_rightmost = None
        max_horizontal_distance = 0  # Initialize the maximum horizontal distance

        # Iterate through points grouped by their y-coordinate (level)
        for level, points in points_by_level_y.items():
            # Sort points by their x-coordinate to find the leftmost and rightmost points on the same level
            points.sort(key=lambda point: point[0])
                
            # Calculate the horizontal distance between the leftmost and rightmost points on this level
            horizontal_distance = round(math.sqrt((points[-1][0] - points[0][0])**2 + (points[-1][1] - points[0][1])**2), 2)

            # If the horizontal distance is greater than the current maximum, update the leftmost and rightmost points
            if horizontal_distance > max_horizontal_distance:
                max_horizontal_distance = horizontal_distance
                same_level_leftmost = points[0]
                same_level_rightmost = points[-1]

        # Calculate the horizontal distance (width) directly from points
        horizontal_distance = max_horizontal_distance

        # Determine which one is the length and which one is the width
        if vertical_distance >= horizontal_distance:
            # Draw a blue diagonal line between topmost and bottommost points
            cv2.line(output_image, topmost_point, bottommost_point, (255, 0, 0), 2)

            # Draw a green line between leftmost and rightmost points on the same level
            cv2.line(output_image, same_level_leftmost, same_level_rightmost, (0, 255, 0), 2)
            
            length = vertical_distance
            width = horizontal_distance
            
            # print("Topmost Point:", topmost_point)
            # print("Bottommost Point:", bottommost_point)
            # print("Leftmost Point:", same_level_leftmost)
            # print("Rightmost Point:", same_level_rightmost)
            
            final_top = topmost_point
            final_bottom = bottommost_point
            final_left = same_level_leftmost
            final_right = same_level_rightmost
                    
        else:
            # Calculate the vertical distance (length) directly from points
            horizontal_distance = round(math.sqrt((leftmost_point[0] - rightmost_point[0])**2 + (leftmost_point[1] - rightmost_point[1])**2), 2)
            
            # Initialize variables to store the topmost and bottommost points on the same x-axis
            same_level_topmost = None
            same_level_bottommost = None
            max_vertical_distance = 0  # Initialize the maximum vertical distance

            # Iterate through points grouped by their x-coordinate (level)
            for level, points in points_by_level_x.items():
                # Sort points by their x-coordinate to find the topmost and bottommost points on the same level
                points.sort(key=lambda point: point[0])
                    
                # Calculate the horizontal distance between the topmost and bottommost points on this level
                vertical_distance = round(math.sqrt((points[-1][0] - points[0][0])**2 + (points[-1][1] - points[0][1])**2), 2)

                # If the horizontal distance is greater than the current maximum, update the topmost and bottommost points
                if vertical_distance > max_vertical_distance:
                    max_vertical_distance = vertical_distance
                    same_level_topmost = points[0]
                    same_level_bottommost = points[-1]

            # Calculate the horizontal distance (width) directly from points
            vertical_distance = max_vertical_distance
            
            # Draw a blue diagonal line between topmost and bottommost points
            cv2.line(output_image, same_level_topmost, same_level_bottommost, (255, 0, 0), 2)

            # Draw a green line between leftmost and rightmost points on the same level
            cv2.line(output_image, leftmost_point, rightmost_point, (0, 255, 0), 2)
            
            length = horizontal_distance
            width = vertical_distance
            
            # print("Topmost Point:", same_level_topmost)
            # print("Bottommost Point:", same_level_bottommost)
            # print("Leftmost Point:", leftmost_point)
            # print("Rightmost Point:", rightmost_point)
            
            final_top = same_level_topmost
            final_bottom = same_level_bottommost
            final_left = leftmost_point
            final_right = rightmost_point
        
        image_output_path = f'static/fish_image_{counter}.png'
        cv2.imwrite(image_output_path, output_image)
        
        # Reference object  
        reference_length_cm = 60
        length_of_reference_in_pixels = 1280

        # Calculate the number of pixels per centimeter
        px_per_cm = length_of_reference_in_pixels / reference_length_cm

        # Convert length and width from pixels to centimeters
        length_cm = length / px_per_cm
        width_cm = width / px_per_cm

        # Convert length and width from centimeters to inches
        length_in = length_cm / 2.54
        width_in = width_cm / 2.54
        
        girthy_co = 0.58
                
        girth = length_in * girthy_co
        weight = length_in * girth * girth / 800
            
        # print("Length:", round(length_in, 2), "In")
        # print("Width:", round(width_in, 2), "In")
        # print("Girth:", round(girth, 2), "In")
        # print("Weight:", round(weight, 4), "Pounds")
        
        add_fish(class_names[counter], final_top, final_bottom, final_left, final_right, round(length_in, 2), round(width_in, 2), round(girth, 2), round(weight, 4), round(accuracies[counter], 2), image_output_path)

        counter += 1
    
    measurements_done = True
    counter = 0
    return recognized_fish

@app.route('/')
def index():
    return render_template('fish/index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file:
        # Save the uploaded file to a folder
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)
        
        if uploaded_file.filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            classes, accuracies = getClassName(model_path, threshold, file_path)
            mask_filenames = getMasks(model_path, file_path, target_width, target_height)
            measured_fish = getFishMeasurements(mask_filenames, classes, accuracies)
            
            measurements = []
            for fish in measured_fish:
                measurement_data = {
                    'Class_Name': fish['class_name'],
                    'Topmost_Point': fish['Topmost_Point'],
                    'Bottommost_Point': fish['Bottommost_Point'],
                    'Leftmost_Point': fish['Leftmost_Point'],
                    'Rightmost_Point': fish['Rightmost_Point'],
                    'length': fish['Length'],
                    'width': fish['Width'],
                    'girth': fish['Girth'],
                    'weight': fish['Weight'],
                    'accuracy': fish['Accuracy'],
                    'Final_Image': fish['Picture']
                }
                measurements.append(measurement_data)
                
            rendered_template = render_template("fish/index.html", measurements=measurements)
            return rendered_template
        
        elif uploaded_file.filename.endswith(('.mp4', '.avi', '.mkv')):
            measured_fish = []
            mask_filenames, classes, accuracies = processVideo(model_path, file_path, target_width, target_height, threshold)
            measured_fish = getFishMeasurements(mask_filenames, classes, accuracies)
            
             # Return the results for videos
            measurements = []
            for fish in measured_fish:
                measurement_data = {
                    'Class_Name': fish['class_name'],
                    'Topmost_Point': fish['Topmost_Point'],
                    'Bottommost_Point': fish['Bottommost_Point'],
                    'Leftmost_Point': fish['Leftmost_Point'],
                    'Rightmost_Point': fish['Rightmost_Point'],
                    'length': fish['Length'],
                    'width': fish['Width'],
                    'girth': fish['Girth'],
                    'weight': fish['Weight'],
                    'accuracy': fish['Accuracy'],
                    'Final_Image': fish['Picture']
                }
                measurements.append(measurement_data)
                                        
            rendered_template = render_template("fish/index.html", measurements=measurements)            
            return rendered_template
                        
if __name__ == '__main__':
    app.run(debug=True)