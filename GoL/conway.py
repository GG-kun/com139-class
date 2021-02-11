"""
conway.py 
A simple Python/matplotlib implementation of Conway's Game of Life.
usage: conway.py [file] [generations]
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

ON = 255
OFF = 0
vals = [ON, OFF]

def fileGrid(configurationFileName):
    """returns a grid of NxM specified by the file with 2D coordinates"""    
    configurationFile = open(configurationFileName, "r")
    lines = configurationFile.readlines()
    if len(lines) <= 0:
        print("file is empty")
        return np.array([])
    
    # First file line is a 2D tuple for the grid dimensions
    dimensions = lines[0].split()
    if not dimensions[0].isnumeric() or not dimensions[1].isnumeric():
        print("at least one file dimension is not numeric")
        return np.array([])
    N = int(dimensions[0])
    M = int(dimensions[1])
    grid = np.zeros(N*M).reshape(N, M)

    # Grid coordinate (0,0) is at top left corner and is the first cell
    for line in lines[1:]:
        coordinates = line.split()
        if not coordinates[0].isnumeric() or not coordinates[1].isnumeric():
            print("at least one file coordinate is not numeric")
            return np.array([])

        # A file coordinate is in the type "y x"
        x = int(coordinates[1])
        y = int(coordinates[0])

        if y >= N or x >= M:
            print("at least a coordinate is outside of the grid")
            return np.array([])
        grid[y][x] = ON

    return grid

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
    # Any live cell with fewer than two live neighbours dies, as if by underpopulation
    # Any live cell with more than three live neighbours dies, as if by overpopulation
    if neighborsCount < 2 or neighborsCount > 3:
        return OFF
    # Any live cell with two or three live neighbours lives on to the next generation
    return cell

def dead_rule(cell, neighborsCount):
    # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction
    if neighborsCount == 3:
        return ON
    # All other dead cells stay dead
    return cell

def update(frameNum, img, grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line 
    newGrid = grid.copy()

    entities = np.array([0,0,0,0,0, # Block, Beehive, Loaf, Boat, Tub
                        0,0,0, # Blinker, Toad, Beacon
                        0,0, # Glider, Light-weight spaceship
                        1]) # Others

    # Grid iteration
    yCells = len(grid)
    for y in range(yCells):
        xCells = len(grid[y])
        for x in range(xCells):
            # Current cell
            cell = grid[y][x]
            # Get current cell neighbors list
            neighbors = []
            for i in range(-1,2):
                for j in range(-1,2):
                    if i == 0 and j == 0:
                        continue
                    neighborY = y+i
                    neighborX = x+j
                    if neighborY >= 0 and neighborY < yCells and neighborX >= 0 and neighborX < xCells:
                        neighbors.append(grid[neighborY][neighborX])
            # Get current cell neighbors count
            neighborsCount = neighbors.count(ON)
            # Live rule on current cell
            if cell == ON:
                cell = live_rule(cell, neighborsCount)
            # Dead rule on current cell  
            else:
                cell = dead_rule(cell, neighborsCount)
            # Update current cell in new grid
            newGrid[y][x] = cell

    f = open("entity_count.txt","a")
    f.write("{0}:".format(frameNum))
    total = sum(entities)
    for entity in entities:
        f.write(" {0}({1}%)".format(entity,entity/total*100))
    f.write("\n")

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

    # set generations number 
    generations = 200

    # set animation update interval
    updateInterval = 1000

    # declare grid
    grid = np.array([])

    # populate grid with random on/off - more off than on
    # default grid when configFile is not given
    grid = randomGrid(N)
    # Uncomment lines to see the "glider" demo
    # grid = np.zeros(N*N).reshape(N, N)
    # addGlider(1, 1, grid)
    # Uncomment lines to see the "block" demo
    # grid = np.zeros(N*N).reshape(N, N)
    # addBlock(1, 1, grid)
    # Uncomment lines to see the "blinker" demo
    # grid = np.zeros(N*N).reshape(N, N)
    # addBlinker(1, 1, grid)

    # Get arguments
    if len(sys.argv) > 1:
        # Get initial configuration file name from arguments
        grid = fileGrid(sys.argv[1])
        if len(grid) <= 0:
            return  
        if len(sys.argv) > 2:
            # Get generations from arguments
            generations = sys.argv[2]
            if generations.isnumeric():
                generations = int(generations)
                if generations <= 0:
                    print("generations must be a higher than 0")
                    return
            else:
                print("usage: conway.py [file] [generations]")
                return

    f = open("entity_count.txt","w")
    f.write("Generation Block Beehive Loaf Boat Tub Blinker Toad Beacon Glider Light-weight spaceship Others\n")

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, ),
                                  frames = generations, repeat = False, # Stop generation after number of generations happen
                                  interval=updateInterval,
                                  save_count=50)

    plt.show()

# call main
if __name__ == '__main__':
    main()