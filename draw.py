import pygame
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 560, 560
PREDICTION_WIDTH = 300
SCREEN = pygame.display.set_mode((WIDTH + PREDICTION_WIDTH, HEIGHT))

FPS = 60
clock = pygame.time.Clock()
frames = 0

cell_size = 20
cells = np.zeros((WIDTH // cell_size, HEIGHT // cell_size))

model = tf.keras.models.load_model("model.keras")
prediction = np.zeros(10)

pygame.font.init()
font = pygame.font.SysFont("Arial", 30)


def get_784_vector():
    image = pygame.surfarray.make_surface(cells)
    image = pygame.transform.scale(image, (28, 28))
    image = pygame.transform.rotate(image, 90)
    image = pygame.transform.flip(image, False, True)
    x = np.array(pygame.surfarray.array2d(image)).reshape(1, 784)
    return x


def main():
    global cells, prediction, frames

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                # Reset
                if event.key == pygame.K_r:
                    cells = np.zeros((WIDTH // cell_size, HEIGHT // cell_size))
                    prediction = np.zeros(10)

                # Append label from input and 1*784 vector to custom_train.csv
                if event.key == pygame.K_s:
                    # Append label
                    label = input("What's the number? ")
                    with open("custom_train.csv", "a") as f:
                        f.write(label)

                    # Append vector
                    x = get_784_vector()
                    with open("custom_train.csv", "a") as f:
                        for i in range(x.shape[1]):
                            f.write("," + str(x[0, i]))
                        f.write("\n")

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            paint_with_adjacent(pos, 255)

        if pygame.mouse.get_pressed()[2]:
            pos = pygame.mouse.get_pos()
            paint_with_adjacent(pos, 0)

        # Predict every second if not empty
        if frames % FPS // 2 == 0 and cells.sum() > 0:
            x = get_784_vector()
            x = x / 255

            y = model.predict(x)
            prediction = softmax(y[0])
            print(np.argmax(y))

        SCREEN.fill((0, 0, 0))
        draw()

        pygame.display.update()
        clock.tick(FPS)
        frames += 1

        pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def paint_with_adjacent(pos, color):
    x, y = pos

    cell_x = x // cell_size
    cell_y = y // cell_size

    if x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT:
        cells[cell_x, cell_y] = color

    if x + cell_size >= 0 and x + cell_size < WIDTH:
        if cells[cell_x + 1, cell_y] != color:
            cells[cell_x + 1, cell_y] = color // 2
    if x - cell_size >= 0 and x - cell_size < WIDTH:
        if cells[cell_x - 1, cell_y] != color:
            cells[cell_x - 1, cell_y] = color // 2
    if y + cell_size >= 0 and y + cell_size < HEIGHT:
        if cells[cell_x, cell_y + 1] != color:
            cells[cell_x, cell_y + 1] = color // 2
    if y - cell_size >= 0 and y - cell_size < HEIGHT:
        if cells[cell_x, cell_y - 1] != color:
            cells[cell_x, cell_y - 1] = color // 2


def draw():
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            color = (cells[i, j], cells[i, j], cells[i, j])
            pygame.draw.rect(SCREEN, color, (i * cell_size, j * cell_size, cell_size, cell_size))

    for i in range(10):
        # Show label
        label = str(i)
        text = font.render(label, True, (255, 255, 255))
        SCREEN.blit(text, (WIDTH + 20, i * 40))

        # Draw red probability bar
        pygame.draw.rect(
            SCREEN,
            (255, 0, 0),
            (
                WIDTH,
                i * 40 + 20,
                PREDICTION_WIDTH * prediction[i],
                20,
            ),
        )


if __name__ == "__main__":
    main()
