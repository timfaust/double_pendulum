import tkinter as tk

import torch
print(torch.cuda.is_available())

root = tk.Tk()
root.title("X11 Forwarding Test")
label = tk.Label(root, text="If you see this, X11 forwarding is working!")
label.pack(padx=20, pady=20)
root.mainloop()

import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((640, 500))
pygame.display.set_caption('Simple Pygame Test')

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((255, 0, 0))
    pygame.display.flip()


