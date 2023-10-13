import pygame
import numpy as np
import nn_scratch

WIDTH, HEIGHT = 560, 560
PREDICTION_WIDTH = 300
SCREEN = pygame.display.set_mode((WIDTH + PREDICTION_WIDTH, HEIGHT))

FPS = 60
clock = pygame.time.Clock()
frames = 0

cell_size = 20
cells = np.zeros((WIDTH // cell_size, HEIGHT // cell_size))

model = nn_scratch.load_model("scratch_model")
y = np.zeros(10)

pygame.font.init()
font = pygame.font.SysFont("Arial", 30)


def get_X():
    image = pygame.surfarray.make_surface(cells)
    image = pygame.transform.scale(image, (28, 28))
    image = pygame.transform.rotate(image, 90)
    image = pygame.transform.flip(image, False, True)
    x = np.array(pygame.surfarray.array2d(image)).reshape(1, 784)
    return x


def main():
    global cells, prediction, frames, y

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    cells = np.zeros((WIDTH // cell_size, HEIGHT // cell_size))

        if pygame.mouse.get_pressed()[0]:
            paint(85)

        if pygame.mouse.get_pressed()[2]:
            paint(-85)

        if frames % FPS == 0 and cells.sum() > 0:
            X = get_X() / 255
            y = model.predict(X)[0]

        SCREEN.fill((0, 0, 0))
        draw()

        pygame.display.update()
        clock.tick(FPS)
        frames += 1

        pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")


def paint(color):
    x, y = pygame.mouse.get_pos()
    cell_x = x // cell_size
    cell_y = y // cell_size
    try:
        paint2(cell_x, cell_y, color)
        paint2(cell_x + 1, cell_y, color)
        paint2(cell_x - 1, cell_y, color)
        paint2(cell_x, cell_y + 1, color)
        paint2(cell_x, cell_y - 1, color)
    except IndexError:
        return


def paint2(x, y, color):
    if cells[x, y] + color <= 255 and cells[x, y] + color >= 0:
        cells[x, y] += color


def draw():
    # Draw cells
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            color = (cells[i, j], cells[i, j], cells[i, j])
            pygame.draw.rect(SCREEN, color, (i * cell_size, j * cell_size, cell_size, cell_size))

    # Draw predictions
    for digit in range(10):
        bar_height = 30
        bar_gap = 10

        # Red bars
        bar_width = y[digit] * PREDICTION_WIDTH
        bar_x = WIDTH + 20
        bar_y = digit * (bar_height + bar_gap) + 10
        pygame.draw.rect(SCREEN, (255, 0, 0), (bar_x, bar_y, bar_width, bar_height))

        # Labels
        label_text = str(digit)
        label = font.render(label_text, True, (255, 255, 255))
        label_x = WIDTH
        label_y = bar_y + (bar_height - font.size(label_text)[1]) // 2
        SCREEN.blit(label, (label_x, label_y))


if __name__ == "__main__":
    main()
