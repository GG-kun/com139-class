"""
Based on the Jos Stam paper https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games
and the mike ash vulgarization https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://github.com/Guilouf/python_realtime_fluidsim
"""
import numpy as np
import math
import json

class Fluid:

    def __init__(self):
        self.rotx = 1
        self.roty = 1
        self.cntx = 1
        self.cnty = -1

        self.size = 60  # map size
        self.dt = 0.2  # time interval
        self.iter = 2  # linear equation solving iteration number

        self.diff = 0.0000  # Diffusion
        self.visc = 0.0000  # viscosity

        self.s = np.full((self.size, self.size), 0, dtype=float)        # Previous density
        self.density = np.full((self.size, self.size), 0, dtype=float)  # Current density

        # array of 2d vectors, [x, y]
        self.velo = np.full((self.size, self.size, 2), 0, dtype=float)
        self.velo0 = np.full((self.size, self.size, 2), 0, dtype=float)

    def step(self):
        self.diffuse(self.velo0, self.velo, self.visc)

        # x0, y0, x, y
        self.project(self.velo0[:, :, 0], self.velo0[:, :, 1], self.velo[:, :, 0], self.velo[:, :, 1])

        self.advect(self.velo[:, :, 0], self.velo0[:, :, 0], self.velo0)
        self.advect(self.velo[:, :, 1], self.velo0[:, :, 1], self.velo0)

        self.project(self.velo[:, :, 0], self.velo[:, :, 1], self.velo0[:, :, 0], self.velo0[:, :, 1])

        self.diffuse(self.s, self.density, self.diff)

        self.advect(self.density, self.s, self.velo)

    def lin_solve(self, x, x0, a, c):
        """Implementation of the Gauss-Seidel relaxation"""
        c_recip = 1 / c

        for iteration in range(0, self.iter):
            # Calculates the interactions with the 4 closest neighbors
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) * c_recip

            self.set_boundaries(x)

    def set_boundaries(self, table):
        """
        Boundaries handling
        :return:
        """

        if len(table.shape) > 2:  # 3d velocity vector array
            # Simulating the bouncing effect of the velocity array
            # vertical, invert if y vector
            table[:, 0, 1] = - table[:, 0, 1]
            table[:, self.size - 1, 1] = - table[:, self.size - 1, 1]

            # horizontal, invert if x vector
            table[0, :, 0] = - table[0, :, 0]
            table[self.size - 1, :, 0] = - table[self.size - 1, :, 0]

        table[0, 0] = 0.5 * (table[1, 0] + table[0, 1])
        table[0, self.size - 1] = 0.5 * (table[1, self.size - 1] + table[0, self.size - 2])
        table[self.size - 1, 0] = 0.5 * (table[self.size - 2, 0] + table[self.size - 1, 1])
        table[self.size - 1, self.size - 1] = 0.5 * table[self.size - 2, self.size - 1] + \
                                              table[self.size - 1, self.size - 2]

    def diffuse(self, x, x0, diff):
        if diff != 0:
            a = self.dt * diff * (self.size - 2) * (self.size - 2)
            self.lin_solve(x, x0, a, 1 + 6 * a)
        else:  # equivalent to lin_solve with a = 0
            x[:, :] = x0[:, :]

    def project(self, velo_x, velo_y, p, div):
        # numpy equivalent to this in a for loop:
        # div[i, j] = -0.5 * (velo_x[i + 1, j] - velo_x[i - 1, j] + velo_y[i, j + 1] - velo_y[i, j - 1]) / self.size
        div[1:-1, 1:-1] = -0.5 * (
                velo_x[2:, 1:-1] - velo_x[:-2, 1:-1] +
                velo_y[1:-1, 2:] - velo_y[1:-1, :-2]) / self.size
        p[:, :] = 0

        self.set_boundaries(div)
        self.set_boundaries(p)
        self.lin_solve(p, div, 1, 6)

        velo_x[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.size
        velo_y[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.size

        self.set_boundaries(self.velo)

    def advect(self, d, d0, velocity):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)

        for j in range(1, self.size - 1):
            for i in range(1, self.size - 1):
                tmp1 = dtx * velocity[i, j, 0]
                tmp2 = dty * velocity[i, j, 1]
                x = i - tmp1
                y = j - tmp2

                if x < 0.5:
                    x = 0.5
                if x > (self.size - 1) - 0.5:
                    x = (self.size - 1) - 0.5
                i0 = math.floor(x)
                i1 = i0 + 1.0

                if y < 0.5:
                    y = 0.5
                if y > (self.size - 1) - 0.5:
                    y = (self.size - 1) - 0.5
                j0 = math.floor(y)
                j1 = j0 + 1.0

                s1 = x - i0
                s0 = 1.0 - s1
                t1 = y - j0
                t0 = 1.0 - t1

                i0i = int(i0)
                i1i = int(i1)
                j0i = int(j0)
                j1i = int(j1)

                try:
                    d[i, j] = s0 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                              s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
                except IndexError:
                    # tmp = str("inline: i0: %d, j0: %d, i1: %d, j1: %d" % (i0, j0, i1, j1))
                    # print("tmp: %s\ntmp1: %s" %(tmp, tmp1))
                    raise IndexError
        self.set_boundaries(d)

    def turn(self):
        self.cntx += 1
        self.cnty += 1
        if self.cntx == 3:
            self.cntx = -1
            self.rotx = 0
        elif self.cntx == 0:
            self.rotx = self.roty * -1
        if self.cnty == 3:
            self.cnty = -1
            self.roty = 0
        elif self.cnty == 0:
            self.roty = self.rotx
        return self.rotx, self.roty

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
        return [self.x, self.y]

    def set_coord(self, array):
        self.x = array[0]
        self.y = array[1]

        return

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
        direction_matrix = np.array(self.direction.get_array())
        
        radians = math.radians(degree)
        spin_matrix = np.array([
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ])

        self.direction.set_coord(np.matmul(direction_matrix, spin_matrix))

    def spray(self, delta_degree, degree):
        direction_matrix = np.array(self.initial_direction.get_array())
        
        delta_degree *= math.sin(math.radians(degree))
        radians = math.radians(delta_degree)
        spin_matrix = np.array([
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ])

        self.direction.set_coord(np.matmul(direction_matrix, spin_matrix))

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
    frames = 120
    current_frame = 0

    def __init__(self, file_path=None):
        if file_path is not None:
            self.set_json(file_path)

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
        
        if "frames" in data:
            self.frames = data["frames"]

    def set_color(self, ax):
        ax.clear()
        ax.contourf(x, y, inst.density, cmap=config.color)

    def set_densities(self, inst):
        for density in self.densities:
            x_limit = density.position.x+density.size.x
            y_limit = density.position.y+density.size.y
            inst.density[density.position.x:x_limit, density.position.y:y_limit] += density.amount

    def set_velocities(self, inst):
        for velocity in self.velocities:
            velocity.animate(self.frames, self.current_frame)
            inst.velo[velocity.position.x, velocity.position.y] = [velocity.direction.x, velocity.direction.y]

        self.current_frame += 1    

    def __repr__(self):
        return "\ncolor:{0}\nvelocities:{1}\ndensities:{2}".format(self.color, self.velocities, self.densities)
    
    def __str__(self):
        return "\ncolor:{0}\nvelocities:{1}\ndensities:{2}".format(self.color, self.velocities, self.densities)


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        inst = Fluid()

        def update_im(i):
            # We add new density creators in here
            config.set_densities(inst)
            # We add velocity vector values in here
            config.set_velocities(inst)
            inst.step()
            im.set_array(inst.density)
            q.set_UVC(inst.velo[:, :, 1], inst.velo[:, :, 0])
            # We add color
            config.set_color(ax)
            # print(f"Density sum: {inst.density.sum()}")
            im.autoscale()

        fig, ax = plt.subplots()
        x = np.linspace(0, inst.size, len(inst.density[:,0]))
        y = np.linspace(0, inst.size, len(inst.density[:,1]))

        config = Config(file_path='config.json')
        # print(config)

        # plot density
        im = plt.imshow(inst.density, vmax=100, interpolation='bilinear')

        # plot vector field
        q = plt.quiver(inst.velo[:, :, 1], inst.velo[:, :, 0], scale=10, angles='xy')
        anim = animation.FuncAnimation(fig, update_im, interval=1, frames=config.frames)
        # anim = animation.FuncAnimation(fig, update_im, interval=0)
        # anim.save("movie.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
        anim.save("movie.gif", fps=30)
        plt.show()

    except ImportError:
        import imageio

        frames = 30

        flu = Fluid()

        video = np.full((frames, flu.size, flu.size), 0, dtype=float)

        for step in range(0, frames):
            flu.density[4:7, 4:7] += 100  # add density into a 3*3 square
            flu.velo[5, 5] += [1, 2]

            flu.step()
            video[step] = flu.density

        imageio.mimsave('./video.gif', video.astype('uint8'))
