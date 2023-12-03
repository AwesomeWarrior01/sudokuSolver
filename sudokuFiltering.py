import cv2
import numpy as np

def count_sudoku_lines(image, distance_threshold=20):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Use Probabilistic Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=150, maxLineGap=15)

    # Draw detected lines on a copy of the original image
    result_image = image.copy()

    # Initialize counters for horizontal and vertical lines
    horizontal_lines = 0
    vertical_lines = 0

    if lines is not None:
        # Merge lines that are close to each other
        lines = merge_close_lines(lines, distance_threshold)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Check if the line is approximately horizontal or vertical
            if abs(y2 - y1) > abs(x2 - x1):
                horizontal_lines += 1
            else:
                vertical_lines += 1

    print("Horizontal Lines:", horizontal_lines)
    print("Vertical Lines:", vertical_lines)

    return result_image, horizontal_lines, vertical_lines

def merge_close_lines(lines, distance_threshold):
    merged_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if the line is close to any existing line in merged_lines
        merged = False
        for merged_line in merged_lines:
            x3, y3, x4, y4 = merged_line[0]

            # Calculate the distance between the endpoints of the two lines
            distance = np.sqrt((x1 - x3)**2 + (y1 - y3)**2) + np.sqrt((x2 - x4)**2 + (y2 - y4)**2)

            if distance < distance_threshold:
                # Merge the lines by updating the endpoints
                merged_line[0] = [min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)]
                merged = True
                break

        if not merged:
            merged_lines.append(line)

    return merged_lines

def display_solution(original_image, solution, sodoku_size, text_size=0.65):
    # Display the solution on the original image
    cell_size = (original_image.shape[1] // sodoku_size[1], original_image.shape[0] // sodoku_size[0])

    for i in range(len(solution)):
        for j in range(len(solution[0])):
            if solution[i][j] != 0:
                x = int((j + 0.4) * cell_size[0])
                y = int((i + 0.7) * cell_size[1])

                # Adjust the text size by changing the scale factor
                font_scale = text_size
                cv2.putText(original_image, str(solution[i][j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

    return original_image

def SudokuFilter():
    # Read the image (INPUT)
    #img = cv2.imread('images/sudoku_medium_1445.jpg') # Used Working
    #img = cv2.imread('images/sudoku_medium_750.jpg') # Used Working
    img = cv2.imread('images/sudoku_medium_750_edit.jpg') # Used Working
    #img = cv2.imread('images/testIMG.jpg') # Used lighter threshold, Working
    #img = cv2.imread('images/news.jpg') # Doesn't work. More filtering would be needed.
    #img = cv2.imread('images/news2.jpg') # Used Darker threshold, Working
    #img = cv2.imread('images/color.png') # Doesn't work

    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a mask
    mask = np.zeros((gray.shape), np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Apply morphological operations
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray) / (close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    # Finding sudoku square and creating a mask image
    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 19, 2)
    contour, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    # Draw contours on the mask
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)

    # Apply the mask
    res = cv2.bitwise_and(res, mask)

    # Find the four corners of the Sudoku puzzle
    pts = cv2.approxPolyDP(best_cnt, 0.02 * cv2.arcLength(best_cnt, True), True)

    # Set width and height
    w, h = 300, 300

    # Order the corners
    try:
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Calculate the perspective transform matrix
    
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)

        # Apply the perspective transformation
        warped = cv2.warpPerspective(img, M, (w, h))

    except:
        warped = cv2.resize(img, (w,h))
    return warped


def Overlay_solution(warped, solution):
    # Count and display Sudoku lines with line merging
    lines_image, horizontal_lines, vertical_lines = count_sudoku_lines(warped)

    # Display the solution on the original image
    solution_displayed_image = display_solution(lines_image, solution, (9, 9))

    # Display the result
    cv2.imshow("Sudoku Lines with Solution", solution_displayed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #define an example solution
    solution = [[9, 0, 1, 4, 3, 5, 2, 6, 8],
                [8, 0, 6, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 2, 0, 6, 0, 7, 1],
                [4, 6, 5, 9, 0, 0, 8, 1, 0],
                [0, 0, 3, 7, 0, 1, 6, 0, 5],
                [0, 9, 0, 0, 5, 8, 0, 0, 4],
                [7, 0, 0, 0, 0, 4, 0, 0, 6],
                [0, 4, 8, 5, 1, 0, 7, 3, 9],
                [5, 0, 9, 8, 0, 7, 0, 0, 2]]
    warped = SudokuFilter()
    Overlay_solution(warped, solution)