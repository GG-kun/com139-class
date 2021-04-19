import numpy as np
import math
import json

class Vector2D:
    x = 0
    y = 0
    
    def __init__(self, vector=None):
        if vector is not None:
            if 'x' in vector:
                self.x = vector['x']
            if 'y' in vector:
                self.y = vector['y']
        
        return
    
    def get_array(self):
        return [self.y, self.x]

    def set_coord(self, array):
        self.x = array[1]
        self.y = array[0]

        return

    def rotate(self, degrees):
        direction_matrix = np.array(self.get_array())
        
        radians = math.radians(degrees)
        spin_matrix = np.array([
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ])

        return np.matmul(direction_matrix, spin_matrix)

    def __repr__(self):
        return "({0},{1})".format(self.x,self.y)
    def __str__(self):
        return "({0},{1})".format(self.x,self.y)

class Velocity:
    position = Vector2D()
    direction = Vector2D()
    initial_direction = Vector2D()
    animation = ""

    def __init__(self, velocity=None):
        if velocity is not None:
            if "position" in velocity:
                self.position = Vector2D(velocity["position"])
            if "direction" in velocity:
                self.direction = Vector2D(velocity["direction"])
                self.initial_direction = Vector2D(velocity["direction"])
            if "animation" in velocity:
                self.animation = velocity["animation"]

        return

    def animate(self, frames, current_frame):
        if self.animation == "spinning":
            self.spin(1.5*360/frames)
        if self.animation == "spraying":
            self.spray(90, 2*current_frame*360/frames)

    def spin(self, degree):
        self.direction.set_coord(self.direction.rotate(degree))

    def spray(self, delta_degree, degree):
        delta_degree *= math.sin(math.radians(degree))
        self.direction.set_coord(self.initial_direction.rotate(delta_degree))

    def __repr__(self):
        return "\nposition:{0}\ndirection:{1}\nanimation:{2}\n".format(self.position, self.direction, self.animation)
    def __str__(self):
        return "\nposition:{0}\ndirection:{1}\nanimation:{2}\n".format(self.position, self.direction, self.animation)

class Density:
    position = Vector2D()
    size = Vector2D()
    amount = 0

    def __init__(self, density=None):
        if density is not None:
            if "position" in density:
                self.position = Vector2D(density["position"])
            if "size" in density:
                self.size = Vector2D(density["size"])
            if "amount" in density:
                self.amount = density["amount"]

        return

    def __repr__(self):
        return "\nposition:{0}\nsize:{1}\namount:{2}\n".format(self.position, self.size, self.amount)
    def __str__(self):
        return "\nposition:{0}\nsize:{1}\namount:{2}\n".format(self.position, self.size, self.amount)    

class Config:
    color = "viridis"
    velocities = []
    densities = []
    objects = []
    frames = 120
    current_frame = 0
    fluid = None

    def __init__(self, file_path=None, fluid=None):
        if file_path is not None:
            self.set_json(file_path)

        if fluid is not None:
            self.fluid = fluid

        return

    def set_json(self, file_path):
        with open(file_path) as f:
            data = json.load(f)

        if "color" in data:
            self.color = data["color"]
        
        if "velocities" in data:
            velocities = data["velocities"]
            for velocity in velocities:
                self.velocities.append(Velocity(velocity))

        if "densities" in data:
            densities = data["densities"]
            for density in densities:
                self.densities.append(Density(density))

        if "objects" in data:
            objects = data["objects"]
            for obj in objects:
                self.objects.append(Density(obj))
        
        if "frames" in data:
            self.frames = data["frames"]

    def set_color(self, ax, x, y):
        ax.clear()
        ax.contourf(x, y, self.fluid.density, cmap=self.color)

    def set_densities(self):
        for density in self.densities:
            x_limit = density.position.x+density.size.x
            y_limit = density.position.y+density.size.y
            self.fluid.density[density.position.y:y_limit, density.position.x:x_limit] += density.amount

    def set_velocities(self):
        for velocity in self.velocities:
            velocity.animate(self.frames, self.current_frame)
            self.fluid.velo[velocity.position.y, velocity.position.x] = [velocity.direction.y, velocity.direction.x]

        self.current_frame += 1    

    def remove_density(self):
        for obj in self.objects:
            x_limit = obj.position.x+obj.size.x
            y_limit = obj.position.y+obj.size.y
            self.fluid.density[obj.position.y:y_limit, obj.position.x:x_limit] = 0

    def set_objects(self):
        for obj in self.objects:
            x_limit = obj.position.x+obj.size.x
            y_limit = obj.position.y+obj.size.y
            self.fluid.density[obj.position.y:y_limit, obj.position.x:x_limit] += obj.amount

    def __repr__(self):
        return "\ncolor:{0}\nvelocities:{1}\ndensities:{2}\nobjects:{3}".format(self.color, self.velocities, self.densities, self.objects)
    
    def __str__(self):
        return "\ncolor:{0}\nvelocities:{1}\ndensities:{2}\nobjects:{3}".format(self.color, self.velocities, self.densities, self.objects)
