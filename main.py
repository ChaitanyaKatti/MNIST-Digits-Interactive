import pygame # For drawing
import numpy as np # For matrix operations
import torch # For model
from torch import nn # For model
import os # For model path

pygame.init()
clock = pygame.time.Clock()

# Make model class
model = nn.Sequential(nn.Linear(784, 500),
                      nn.ReLU(),
                      nn.Linear(500, 100),
                      nn.ReLU(),
                      nn.Linear(100, 10),
                      nn.Softmax(dim=1))

# Load from trained model
model.load_state_dict(torch.load(os.getcwd() + '\mnist_model.pth'))

# Screen
screen = pygame.display.set_mode((1920, 1080))
screen.fill((30, 30, 30))
x, y = screen.get_size()

# Drawing Canvas
canvasSize = (840, 840)
canvas = pygame.Surface(canvasSize)
canvas_pos = (x - canvasSize[0] - 80, y/2 - canvasSize[1]/2)
screen.blit(canvas, canvas_pos)

# Prediction Canvas
prediction_canvas = pygame.Surface(canvasSize)
prediction_canvas_pos = (80, y/2 - canvasSize[1]/2)
screen.blit(prediction_canvas, prediction_canvas_pos)

# Draw instruction text
font = pygame.font.SysFont(None, 50)
instruction_text = font.render('Press Backspace to clear canvas', True, (255, 255, 255))
screen.blit(instruction_text, (1170,1000))
instruction_text = font.render('Press ESC to exit', True, (255, 255, 255))
screen.blit(instruction_text, (340,1000))


# Draw rectangele for each digit
def draw_prediction(predictions):
    x_coord = [130 + 60*i for i in range(10)]
    SCLAING_FACTOR = 500
    font = pygame.font.SysFont(None, 50)
    prediction_canvas.fill((0, 0, 0))
    
    for i in range(len(x_coord)):
        pygame.draw.rect(prediction_canvas, (255, 255*(1 - predictions[i]), 255*(1 - predictions[i])),
                         [x_coord[i], 780 - SCLAING_FACTOR*predictions[i], 40, SCLAING_FACTOR*predictions[i]])
    screen.blit(prediction_canvas, prediction_canvas_pos)
    
    for i in range(len(x_coord)):
        img = font.render(str(i), True, (255, 255, 255))
        screen.blit(img, (90 + x_coord[i], 910))
    pygame.display.update()

# Brush
brush_size = 30
brush = pygame.Surface((3*brush_size, 3*brush_size), pygame.SRCALPHA)

# Create a 2d array of size 28x28 to store the grayscale values of the digit
grayscale = np.zeros((28, 28))

# Create a brush matrix, 3x3 grid, opacity values b/w 0 and 1
brush_opacity = np.zeros((3, 3))
brush_opacity[1, 1] = 0.8  # Center
brush_opacity[[0, 1, 1, 2], [1, 0, 2, 1]] = 0.6  # Up, Left, Right, Down
brush_opacity[[0, 0, 2, 2], [0, 2, 0, 2]] = 0.3 # Up Left, Up Right, Down Left, Down Right

# Create a brush
pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[1, 1]),
                 [brush_size, brush_size, brush_size, brush_size])  # Center

pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[2, 1]),
                 [brush_size, 2*brush_size, brush_size, brush_size])  # Down
pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[0, 1]), 
                 [brush_size, 0, brush_size, brush_size])  # Up

pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[1, 2]), 
                 [2*brush_size, brush_size, brush_size, brush_size])  # Right
pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[1, 0]), 
                 [0, brush_size, brush_size, brush_size])  # Left

pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[2, 2]), 
                 [2*brush_size, 2*brush_size, brush_size, brush_size])  # Down Right
pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[2, 0]), 
                 [0, 2*brush_size, brush_size, brush_size])  # Down Left
pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[0, 2]), 
                 [2*brush_size, 0,brush_size, brush_size])  # Up Right
pygame.draw.rect(brush, (255, 255, 255, 255*brush_opacity[0, 0]), 
                 [0,0,brush_size, brush_size])  # Up Left


mouse_down = False
prev_rect = None
while True:
    clock.tick(60)

    for event in pygame.event.get(): # When user presses a button or moves mouse
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
            prev_rect = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                canvas.fill((0, 0, 0))
                screen.blit(canvas, canvas_pos)
                grayscale = np.zeros((28, 28))
            elif event.key == pygame.K_ESCAPE:
                exit()

        if mouse_down: # If mouse is pressed
            mouse_pos = pygame.mouse.get_pos()
            mouse_pos = (mouse_pos[0] - canvas_pos[0],
                         mouse_pos[1] - canvas_pos[1]) # Get mouse position relative to canvas
            rect_center = (np.floor(mouse_pos[0]/brush_size)*brush_size + brush_size/2, np.floor(
                mouse_pos[1]/brush_size)*brush_size + brush_size/2) # Convert mouse pos to 28x28 grid coordinates
            
            # If mouse is outside canvas
            if rect_center[0] < brush_size or rect_center[0] >= canvasSize[0] - brush_size or rect_center[1] < brush_size or rect_center[1] >= canvasSize[1] - brush_size:
                continue
            # If mouse is in same grid as previous
            if prev_rect == rect_center:
                continue
            
            # Set previous rect to current rect
            prev_rect = rect_center
            brush_rect = brush.get_rect(center=rect_center)

            # Draw brush on canvas and draw canvas to screen
            canvas.blit(brush, brush_rect)
            screen.blit(canvas, canvas_pos)

            # Update grayscale
            rect_index = (int(np.floor(mouse_pos[0]/brush_size)), 
                          int(np.floor(mouse_pos[1]/brush_size)))
            
            grayscale[rect_index[1] - 1: rect_index[1] + 2,
                      rect_index[0] - 1: rect_index[0] + 2] += brush_opacity

    # Make predictions, grayscale is 28x28 float64 numpy array
    # Convert it to a 1x784 float32 tensor of shape (1, 784)
    # Then pass it to the model and get predictions
    # Convert model output to numpy array and draw predictions
    predictions = model(torch.from_numpy(grayscale).float().flatten().unsqueeze(0)).squeeze().detach().numpy()
    draw_prediction(predictions)
    
    pygame.display.update()

# thx copilot :)