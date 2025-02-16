import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Snake variables
snake_pos = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]
direction = 'RIGHT'
change_to = direction
speed = 10
score = 0
font = pygame.font.Font(None, 36)
game_over = False

# Food variables
def generate_food():
    while True:
        food = [random.randrange(1, WIDTH//10) * 10, random.randrange(1, HEIGHT//10) * 10]
        if food not in snake_body:
            return food

food_pos = generate_food()
food_spawn = True

# OpenCV camera setup
cap = cv2.VideoCapture(0)
lock = threading.Lock()
frame_surface = pygame.Surface((WIDTH, HEIGHT))

# Hand tracking thread
def process_hand_tracking():
    global change_to, frame_surface, game_over
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                index_finger = landmarks[8]
                thumb = landmarks[4]
                x, y = int(index_finger.x * WIDTH), int(index_finger.y * HEIGHT)
                
                if game_over and abs(index_finger.x - thumb.x) < 0.05 and abs(index_finger.y - thumb.y) < 0.05:
                    reset_game()
                
                if x < WIDTH // 3:
                    change_to = 'LEFT'
                elif x > 2 * WIDTH // 3:
                    change_to = 'RIGHT'
                elif y < HEIGHT // 3:
                    change_to = 'UP'
                elif y > 2 * HEIGHT // 3:
                    change_to = 'DOWN'
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        
        with lock:
            frame_surface = frame

hand_tracking_thread = threading.Thread(target=process_hand_tracking, daemon=True)
hand_tracking_thread.start()

def reset_game():
    global snake_pos, snake_body, direction, change_to, score, speed, food_pos, game_over
    snake_pos = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50]]
    direction = 'RIGHT'
    change_to = direction
    score = 0
    speed = 10
    food_pos = generate_food()
    game_over = False

running = True
while running:
    screen.fill(BLACK)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    with lock:
        screen.blit(frame_surface, (0, 0))
    
    if game_over:
        game_over_text = font.render(f'Game Over! Score: {score}', True, WHITE)
        restart_text = font.render('Pinch to Restart', True, WHITE)
        screen.blit(game_over_text, (WIDTH//4, HEIGHT//3))
        screen.blit(restart_text, (WIDTH//3, HEIGHT//2))
    else:
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
        
        if direction == 'UP':
            snake_pos[1] -= 10
        if direction == 'DOWN':
            snake_pos[1] += 10
        if direction == 'LEFT':
            snake_pos[0] -= 10
        if direction == 'RIGHT':
            snake_pos[0] += 10
        
        # Boundary Wrapping Mode
        snake_pos[0] %= WIDTH
        snake_pos[1] %= HEIGHT
        
        snake_body.insert(0, list(snake_pos))
        if snake_pos == food_pos:
            score += 1
            speed += 1
            food_pos = generate_food()
        else:
            snake_body.pop()
        
        if snake_pos in snake_body[1:]:
            game_over = True
        
        for pos in snake_body:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
        
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, [10, 10])
    
    pygame.display.update()
    clock.tick(speed)

pygame.quit()
cap.release()
cv2.destroyAllWindows()
