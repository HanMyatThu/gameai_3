# sound.py
import pygame

pygame.mixer.init()

sounds = {
    "jump": pygame.mixer.Sound("assets/sounds/jump.wav"),
    "score": pygame.mixer.Sound("assets/sounds/score.wav"),
    "hit": pygame.mixer.Sound("assets/sounds/hit.mp3"),
    "background": pygame.mixer.Sound("assets/sounds/background.mp3"),
    "night": pygame.mixer.Sound("assets/sounds/night.mp3"),
    "day": pygame.mixer.Sound("assets/sounds/day.mp3"),
    "space": pygame.mixer.Sound("assets/sounds/space.mp3"),
    "hell": pygame.mixer.Sound("assets/sounds/hell.mp3"),
    "next": pygame.mixer.Sound("assets/sounds/nextlevel.mp3"),
}

def play(sound_name):
    if sound_name in sounds:
        sounds[sound_name].play()

def stop(sound_name):
    if sound_name in sounds:
        sounds[sound_name].stop()