import cv2
import numpy as np
import heapq


def loadImageAsGrid(image_path):

    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_coords = cv2.findNonZero(green_mask)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_coords = cv2.findNonZero(blue_mask)

    if green_coords is None or blue_coords is None:
        raise ValueError(" Start or goal point not found!")

    image[np.where(green_mask > 0)] = (255, 255, 255)
    image[np.where(blue_mask > 0)] = (255, 255, 255)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

    maze = np.zeros_like(binary_image, dtype=np.uint8)
    for r in range(binary_image.shape[0]):
        for c in range(binary_image.shape[1]):
            if binary_image[r][c] != 255:
                maze[r][c] = 1
            else:
                maze[r][c] = 0

    start = tuple(map(int, green_coords.mean(axis=0)[0]))  # (x, y)
    goal = tuple(map(int, blue_coords.mean(axis=0)[0]))
    start = (start[1], start[0])  # row, col
    goal = (goal[1], goal[0])

    return maze, start, goal

def manhattanDistance(point1, point2):
  r1 = point1[0]
  r2 = point2[0]
  c1 = point1[1]
  c2 = point2[1]
  val = np.abs(r1 - r2) + np.abs(c1 - c2)
  return val

def astarPathfinding(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    g_score = {start: 0}
    f_score = {start: manhattanDistance(start, goal)}

    heapq.heappush(open_list, (f_score[start], start, [start]))

    while open_list:
        _, current, path = heapq.heappop(open_list)

        if current == goal:
            return path  # Found path

        row, col = current
        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr, nc] == 1:
                continue

            tentative_g = g_score[current] + 1

            neighbor = (nr, nc)

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattanDistance(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor, path + [neighbor]))

    return None

imgpath = r"C:\Users\drsen\Downloads\WhatsApp Image 2025-10-23 at 7.19.40 PM.jpeg"
grid, start, goal = loadImageAsGrid(imgpath)
print("Start:", start)
print("Goal:", goal)

path = astarPathfinding(grid, start, goal)

image = cv2.imread(imgpath)


if path:
    for i in range(1, len(path)):
        y1, x1 = path[i-1]
        y2, x2 = path[i]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.circle(image, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.circle(image, (goal[1], goal[0]), 5, (255, 0, 0), -1)
    cv2.imshow("solvedimg",image)
    cv2.imshow("solvedimg",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("maze_path_output.png", image)
    print("Path length :",len(path))

else:
    print("No path found")