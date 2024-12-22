from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random

data_collection = True
width = 1536
height = 864
refresh_rate = 60.02
stim_duration = 1.2

def create_32_targets(size=120, colors=[-1, -1, -1] * 32, checkered=False):
    positions = []
    positions.extend([[-width / 2 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 1 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 2 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 3 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 4 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 5 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 6 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    positions.extend([[-width / 2 + 190 * 7 + 100, height / 2 - 90 - i * 200 - 80] for i in range(4)])
    if checkered:
        texture = checkered_texure()
    else:
        texture = None
    keys = visual.ElementArrayStim(window, nElements=32, elementTex=texture, elementMask=None, units='pix',
                                   sizes=[size, size], xys=positions, colors=colors)
    return keys

def checkered_texure():
    rows = 8  # Replace with desired number of rows
    cols = 8  # Replace with desired number of columns
    array = np.zeros((rows, cols))
    for i in range(rows):
        array[i, ::2] = i % 2  # Set every other element to 0 or 1, alternating by row
        array[i, 1::2] = (i+1) % 2  # Set every other element to 0 or 1, alternating by row
    return np.kron(array, np.ones((16, 16)))*2-1

def create_trial_sequence(n_per_class, classes=[(7.5, 0), (8.57, 0), (10, 0), (12, 0), (15, 0)], seed=0):
    """
    Create a random sequence of trials with n_per_class of each class
    Inputs:
        n_per_class : number of trials for each class
    Outputs:
        seq : (list of len(10 * n_per_class)) the trial sequence
    """
    seq = classes * n_per_class
    random.seed(seed)
    random.shuffle(seq)  # shuffles in-place
    return seq

keyboard = keyboard.Keyboard()
window = visual.Window(
        size = [width,height],
        checkTiming = True,
        allowGUI = False,
        fullscr = True,
        useRetina = False,
    )
visual_stimulus = create_32_targets(checkered=True)
num_frames = np.round(stim_duration * refresh_rate).astype(int)  # total number of frames per trial
frame_indices = np.arange(num_frames)  # frame indices for the trial
stimulus_classes = [(8, 0), (8, 0.5), (8, 1), (8, 1.5),
                    (9, 0), (9, 0.5), (9, 1), (9, 1.5),
                    (10, 0), (10, 0.5), (10, 1), (10, 1.5),
                    (11, 0), (11, 0.5), (11, 1), (11, 1.5),
                    (12, 0), (12, 0.5), (12, 1), (12, 1.5),
                    (13, 0), (13, 0.5), (13, 1), (13, 1.5),
                    (14, 0), (14, 0.5), (14, 1), (14, 1.5),
                    (15, 0), (15, 0.5), (15, 1), (15, 1.5), ]
stimulus_frames = np.zeros((num_frames, len(stimulus_classes)))
for i_class, (flickering_freq, phase_offset) in enumerate(stimulus_classes):
        phase_offset += .00001  # nudge phase slightly from points of sudden jumps for offsets that are pi multiples
        stimulus_frames[:, i_class] = signal.square(2 * np.pi * flickering_freq * (frame_indices / refresh_rate) + phase_offset * np.pi)  # frequency approximation formula
trial_sequence = create_trial_sequence(n_per_class=1, classes=stimulus_classes, seed=0)


if data_collection:
    for i_trial, (flickering_freq, phase_offset) in enumerate(trial_sequence):
        keys = keyboard.getKeys()
        if 'escape' in keys:
            core.quit()
        for i_frame in range(num_frames):
            visual_stimulus.colors = np.array([stimulus_frames[i_frame]] * 3).T
            visual_stimulus.draw()
            window.flip()
            if i_frame == num_frames - 1:
                core.wait(1)  # wait for 1 second after the last frame
        core.wait(1)  # wait for 1 second after the last trial
else:
    while True:
        keys = keyboard.getKeys()
        if 'escape' in keys:
            core.quit()
        for i_frame in range(num_frames):
            visual_stimulus.colors = np.array([stimulus_frames[i_frame]] * 3).T
            visual_stimulus.draw()
            window.flip()
            if i_frame == num_frames - 1:
                core.wait(1)  # wait for 1 second after the last frame
        core.wait(1)  # wait for 1 second after the last trial