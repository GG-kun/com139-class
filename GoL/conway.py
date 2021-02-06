"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

ON = 255
OFF = 0
vals = [ON, OFF]

def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)

def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0,    0, 255], 
                       [255,  0, 255], 
                       [0,  255, 255]])
    grid[i:i+3, j:j+3] = glider

def addBlock(i, j, grid):
    """adds a block with top left cell at (i, j)"""
    block = np.array([[255, 255], 
                        [255, 255]])
    grid[i:i+2, j:j+2] = block

def addBlinker(i, j, grid):
    """adds a blinker with top left cell at (i, j)"""
    blinker = np.array([[0,    255, 0], 
                        [0,  255, 0], 
                        [0,  255, 0]])
    grid[i:i+3, j:j+3] = blinker

def live_rule(cell, neighborsCount):
    if neighborsCount < 2 or neighborsCount > 3:
        return OFF
    return cell

def dead_rule(cell, neighborsCount):
    if neighborsCount == 3:
        return ON
    return cell

def update(frameNum, img, grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line 
    newGrid = grid.copy()
    # TODO: Implement the rules of Conway's Game of Life
    yCells = len(grid)
    for y in range(yCells):
        xCells = len(grid[y])
        for x in range(xCells):
            cell = grid[y][x]
            # Get neighbors list
            neighbors = []
            for i in range(-1,2):
                for j in range(-1,2):
                    if i == 0 and j == 0:
                        continue
                    neighborY = y+i
                    neighborX = x+j
                    if neighborY >= 0 and neighborY < yCells and neighborX >= 0 and neighborX < xCells:
                        neighbors.append(grid[neighborY][neighborX])
            # Get neighbors count
            neighborsCount = neighbors.count(ON)
            # Live rule
            if cell == ON:
                cell = live_rule(cell, neighborsCount)
            # Dead rule    
            else:
                cell = dead_rule(cell, neighborsCount)
            newGrid[y][x] = cell

    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,

# main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life system.py.")
    
    # set grid size
    N = 100
    if len(sys.argv) > 1:
        N = sys.argv[1]
        if N.isnumeric():
            N = int(N)
        else:
            print("usage: conway.py [size] [generation] [file]")
            return
        if len(sys.argv) > 2:
            configurationFile = sys.argv[2]    
        
    # set animation update interval
    updateInterval = 50

    # declare grid
    grid = np.array([])
    # populate grid with random on/off - more off than on
    # grid = randomGrid(N)
    # Uncomment lines to see the "glider" demo
    # grid = np.zeros(N*N).reshape(N, N)
    # addGlider(1, 1, grid)
    # Uncomment lines to see the "block" demo
    # grid = np.zeros(N*N).reshape(N, N)
    # addBlock(1, 1, grid)
    # Uncomment lines to see the "blinker" demo
    # grid = np.zeros(N*N).reshape(N, N)
    # addBlinker(1, 1, grid)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, ),
                                  frames = 10,
                                  interval=updateInterval,
                                  save_count=50)

    plt.show()

# call main
if __name__ == '__main__':
    main()