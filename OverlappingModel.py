"""
Author: Jacob Trzcinski
Advisor: Dr. Matteo Luisi
Capstone I
Generating Novel Chiplet Designs Utilizing
the Wave-Function Collapse Algorithm in Pyton

The purpose of this program is to input a bitmap
Split the bitmap into chunks,
Generate the Entropy of each relative piece,
rotate and reflect each piece to see if it is 
symmetric and see the propagations of it
creates weights / entropy array
"""
import time
import math
import random
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import uuid
import collections

hackstring = "" # these will help us name our outputs
hackcount = 0   # they need to be global

class Model:
    """
    Makes an instance of the blank output image
    """
    ### this class makes an instance of the output image,
    ### try not to get hung up on values because most of them will change in child instances

    def __init__(self, width, height):  
        
        ### initialize the model, make our blank output image ###

        self.periodic = False
        self.image_width = width
        self.image_height = height
        self.T = 2

        self.rng = random.Random()

        self.wave = [[[False for _ in range(self.T)] for _ in range(self.image_height)] for _ in range(self.image_width)]   # 3d list for the wavefunction

        self.observe_count = 0
        self.count_prop_passes = 0
        self.changes = [[False for _ in range(self.image_height)] for _ in range(self.image_width)]

        self.observed = None

        ### some stats stuff for us to use in observing ###

        self.log_prob = 0
        self.log_t = math.log(self.T)

        self.stationary = []

    def Clear(self):    # Clears output
        for x in range(0,self.image_width):
            for y in range(0,self.image_height):
                for t in range(0,self.T):   # go thru the entire wave array
                    self.wave[x][y][t] = True
                self.changes[x][y] = False

    def Propogate(self):    # remove any propogated data from the output
        change = False
        b = False

        for x1 in range(0, self.image_width):
            for y1 in range(0, self.image_height):
                if (self.changes[x1][y1]):
                    self.changes[x1][y1] = False
                    dx = (0 - self.N) + 1

                    while dx < self.N:
                        dy = (0 - self.N) + 1

                        while dy < self.N:
                            x2 = x1 + dx
                            y2 = y1 + dy

                            if x2 < 0:
                                x2 += self.image_width
                            elif x2 >= self.image_width:
                                    x2 -= self.image_width

                            if y2 < 0:
                                y2 += self.image_height
                            elif y2 >= self.image_height:
                                    y2 -= self.image_height

                            if (not self.periodic) and (x2 + self.N > self.image_width or y2 + self.N > self.image_height):
                                pass
                            else:
                                w1 = self.wave[x1][y1]
                                w2 = self.wave[x2][y2]

                                p = self.propagator[(self.N - 1) - dx][(self.N - 1) - dy]

                                for t2 in range(0, self.T):
                                    if (not w2[t2]):
                                        pass
                                    else:
                                        b = False
                                        prop = p[t2]
                                        i = 0
                                        while (i < len(prop)) and (b == False):
                                            b = w1[prop[i]]
                                            i += 1
                                        if b == False:
                                            self.changes[x2][y2] = True
                                            change = True
                                            w2[t2] = False
                            dy += 1
                        dx += 1
        return change

    def Graphics(self):
        return Image.new("RGB", (self.image_width, self.image_height),(0,0,0))

    def OnBoundary(self, x, y):
        return True

    def Observe(self):
        """
        Observe a piece from the lowest entropy tile
        """
        self.observe_count += 1
        observed_min = 1e+3
        observed_sum = 0
        main_sum = 0
        log_sum = 0
        noise = 0
        entropy = 0

        minx = -1
        miny = -1
        amount = None
        w = []

        ### Find the minimum entropy for each piece ###
        for x in range(0, self.image_width):
            for y in range(0, self.image_height):
                if self.OnBoundary(x, y): # check if a piece is on the boundary, then skip it anyway
                    pass
                else:
                    w = self.wave[x][y] # copy our wave x and y
                    amount = 0
                    observed_sum = 0
                    t = 0
                    while t < self.T:
                        if w[t]:
                            amount += 1
                            observed_sum += self.stationary[t]
                        t += 1
                    if observed_sum == 0:
                        return False    # we are saying that nothing is observed
                    noise = 1e-6 * self.rng.random()
                    if amount == 1:
                        entropy = 0
                    elif amount == self.T:
                        entropy = self.log_t
                    else:
                        main_sum = 0
                        log_sum = math.log(observed_sum)
                        t = 0
                        while t < self.T:
                            if w[t]:
                                main_sum += self.stationary[t] * self.log_prob[t]
                            t += 1
                        entropy = log_sum - main_sum / observed_sum
                    if entropy > 0 and (entropy + noise < observed_min):
                        observed_min = entropy + noise
                        minx = x
                        miny = y

        ### if there is no min entropy, mark everything as observed ###
        if (-1 == minx) and (-1 == miny):
            # set whole observed array to 0
            self.observed = [[0 for _ in range(self.image_height)] for _ in range(self.image_width)]

            # check if the wave array has the same size
            # if it does, set observed to that ever the t of the wave
            for x in range(0, self.image_width):
                self.observed[x] = [0 for _ in range(self.image_height)]
                for y in range(0, self.image_height):
                    for t in range(0, self.T):
                        if self.wave[x][y][t]:
                            self.observed[x][y] = t
                            break
            return True    # we are saying that it the image is done

        dist = [0 for _ in range(0,self.T)]
        for t in range(0,self.T):
            dist[t] = self.stationary[t] if self.wave[minx][miny][t] else 0
            # turns out you can make an if statement 1 line long

        r = StuffRandom(dist, self.rng.random())
        for t in range(0,self.T):
            self.wave[minx][miny][t] = (t == r)
        self.changes[minx][miny] = True
        return None

    def Run(self, seed, limit):
        ### alright, lets give this Model class some stuff to do ###

        self.log_t = math.log(self.T)
        self.log_prob = [0 for _ in range(self.T)]
        for t in range(0,self.T):
            self.log_prob[t] = math.log(self.stationary[t])

        self.Clear()    # clear the output image

        self.rng = random.Random()  # idk why I have to re-declare this here
                                    # but VSCode doesnt like it when I dont

        self.rng.seed(seed)
        l = 0
        while (1 < limit) or (0 == limit):
            l += 1
            result = self.Observe() # observe a tile on the output
            if None != result:
                return result
            pcount = 0
            presult = True
            global hackcount

            while (presult):
                presult = self.Propogate()

                hackcount += 1

                pcount += 1
        
        #self.Graphics().save("in_progress_{0}_{1}.png".format(hackstring, hackcount), format="PNG")
        return True



class OverlappingModel(Model):
    """
    This class takes in a blank wavefunction, and assembles it from a sample image
    """
    def __init__(self, width, height, name, N_value, symmetry_value, ground_value, high_score):
        """
        initializes the blank model
        """
        super(OverlappingModel, self).__init__(width, height)
        self.N = N_value
        self.periodic = False
        self.bitmap = Image.open("Samples/{0}.png".format(name))
        self.SMX = self.bitmap.size[0]
        self.SMY = self.bitmap.size[1]

        # sample is an array that holds index values for colors in the sample image
        self.sample = [[0 for _ in range(self.SMY)] for _ in range(self.SMX)]
        # colors lists the colors found in the sample image
        self.colors = []

        self.propagator = [[[[]]]]

        for y in range(0, self.SMY):
            for x in range(0, self.SMX):
                a_color = self.bitmap.getpixel((x, y))
                color_exists = [c for c in self.colors if c == a_color]
                if len(color_exists) < 1:
                        self.colors.append(a_color)
                sample_result = [i for i,v in enumerate(self.colors) if v == a_color]
                self.sample[x][y] = sample_result

        self.color_count = len(self.colors)
        self.W = StuffPower(self.color_count, self.N * self.N)

        self.patterns = [[]]

        def FuncPattern(fn):
            result = [0 for _ in range(self.N * self.N)]
            for y in range(0, self.N):
                for x in range(0, self.N):
                    result[x + (y * self.N)] = fn(x, y)
            return result

        pattern_fn = FuncPattern

        def PatternFromSample(x, y):
            def InnerPattern(dx, dy):
                return self.sample[(x + dx) % self.SMX][(y + dy) % self.SMY]
            return pattern_fn(InnerPattern)

        def Rotate(p):
            return FuncPattern(lambda x, y: p[self.N - 1 - y + x * self.N])

        def Reflect(p):
            return FuncPattern(lambda x, y: p[self.N - 1 - x + y * self.N])

        def Index(p):
            result = 0
            power = 1
            for i in range(0, len(p)):
                result = result + (sum(p[len(p) - 1 - i]) * power)
                power = power * self.color_count
            return result
        
        def PatternFromIndex(ind):
            residue = ind
            power = self.W
            result = [None for _ in range(self.N * self.N)]
            for i in range(0, len(result)):
                power = power / self.color_count
                count = 0
                while residue >= power:
                    residue = residue - power
                    count += 1
                result[i] = count
            return result
        
        self.weights = collections.Counter()
        ordering = []

        ylimit = self.SMY - self.N + 1
        xlimit = self.SMX - self.N + 1

        # print(PatternFromSample(20,30))

        for y in range(0, ylimit):
            for x in range(0, xlimit):
                ps = [0 for _ in range(8)]  # In the reference code this looks like [[0],[0],[0],[0],[0],[0],[0],[0]], x8], while my code looks like [0,0,0,0,0,0,0,0],x8
                ps[0] = PatternFromSample(x,y)
                ps[1] = Reflect(ps[0])
                ps[2] = Rotate(ps[0])
                ps[3] = Reflect(ps[2])
                ps[4] = Rotate(ps[2])
                ps[5] = Reflect(ps[4])
                ps[6] = Rotate(ps[4])
                ps[7] = Reflect(ps[6])
                for k in range(0,symmetry_value):
                    ind = Index(ps[k])
                    indexed_weight = collections.Counter({ind : 1})
                    self.weights = self.weights + indexed_weight
                    if not ind in ordering:
                        ordering.append(ind)
 
        self.T = len(self.weights)
        self.ground = int((ground_value + self.T) % self.T)

        self.patterns = [[None] for _ in range(self.T)]
        self.stationary = [None for _ in range(self.T)]
        self.propagator = [[[[0]]] for _ in range(2 * self.N - 1)]

        counter = 0
        for w in ordering:
            self.patterns[counter] = PatternFromIndex(w)
            self.stationary[counter] = self.weights[w]
            counter += 1

        for x in range(0, self.image_width):
            for y in range(0, self.image_height):
                self.wave[x][y] = [False for _ in range(self.T)]

        def Agrees(p1, p2, dx, dy):
            #print(dx)
            #print(dy)
            agreesBool = True
            x_min = dx
            x_max = self.N
            y_min = dy
            y_max = self.N

            if dx < 0:
                x_min = 0
                x_max = dx + self.N
            if dy < 0:
                y_min = 0
                y_max = dy + self.N
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if p1[x + self.N * y] != p2[x - dx + self.N * (y - dy)]:
                        #print(p1[x + self.N * y] != p2[x - dx + self.N * (y - dy)])
                        agreesBool = False
            return agreesBool
            
        for x in range(0, 2 * self.N - 1):
            self.propagator[x] = [[[0]] for _ in range(2 * self.N - 1)]
            for y in range(0, 2 * self.N - 1):
                self.propagator[x][y] = [[0] for _ in range(self.T)]

                for t in range(0, self.T):
                    agreesList = []
                    for t2 in range(0, self.T):
                        if Agrees(self.patterns[t], self.patterns[t2], x - self.N + 1, y - self.N + 1):
                            agreesList.append(t2)
                    self.propagator[x][y][t] = [0 for _ in range(len(agreesList))]
                    for c in range(0, len(agreesList)):
                        self.propagator[x][y][t][c] = agreesList[c]
        return



    def OnBoundary(self, x, y):
        return (not self.periodic) and ((x + self.N > self.image_width) or (y + self.N > self.image_height))

    def Propagate(self):
        change = False
        b = False

        for x1 in range(0, self.image_width):
            for y1 in range(0, self.image_height):
                if (self.changes[x1][y1]):
                    self.changes[x1][y1] = False
                    dx = (0 - self.N) + 1

                    while dx < self.N:
                        dy = (0 - self.N) + 1

                        while dy < self.N:
                            x2 = x1 + dx
                            y2 = y1 + dy

                            if x2 < 0:
                                x2 += self.image_width
                            elif x2 >= self.image_width:
                                    x2 -= self.image_width

                            if y2 < 0:
                                y2 += self.image_height
                            elif y2 >= self.image_height:
                                    y2 -= self.image_height

                            if (not self.periodic) and (x2 + self.N > self.image_width or y2 + self.N > self.image_height):
                                pass
                            else:
                                w1 = self.wave[x1][y1]
                                w2 = self.wave[x2][y2]

                                p = self.propagator[(self.N - 1) - dx][(self.N - 1) - dy]

                                for t2 in range(0, self.T):
                                    if (not w2[t2]):
                                        pass
                                    else:
                                        b = False
                                        prop = p[t2]
                                        i = 0
                                        while (i < len(prop)) and (b == False):
                                            b = w1[prop[i]]
                                            i += 1
                                        if b == False:
                                            self.changes[x2][y2] = True
                                            change = True
                                            w2[t2] = False
                            dy += 1
                        dx += 1
        return change

    def Graphics(self):
        result = Image.new("RGB", (self.image_width, self.image_height),(0,0,0))
        bitmap_data = list(result.getdata())
        if(self.observed != None):
            for y in range(0, self.image_height):
                dy = self.N - 1
                if (y < (self.image_height - self.N + 1)):
                    dy = 0
                for x in range(0, self.image_width):
                    dx = 0
                    if (x < (self.image_width - self.N + 1)):
                        dx = self.N - 1
                    local_observed = self.observed[x - dx][y - dy]
                    local_pattern = self.patterns[local_observed][dx + dy * self.N]
                    c = self.colors[local_pattern]
                    if isinstance(c, (int, float)):
                        bitmap_data[x + y * self.image_width] = (c, c, c)
                    else:
                        bitmap_data[x + y * self.image_width] = (c[0], c[1], c[2])

        else:
            for y in range(0, self.image_height):
                for x in range(0, self.image_width):
                    contributors = 0
                    r = 0
                    g = 0
                    b = 0
                    for dy in range(0, self.N):
                        for dx in range(0, self.N):
                            sx = x - dx
                            sy = y - dy

                            if sx < 0:
                                sx += self.image_width
                            if sy < 0:
                                sy += self.image_height

                            if (self.OnBoundary(sx, sy)):
                                pass
                            else:
                                for t in range(0, self.T):
                                    if self.wave[sx][sy][t]:
                                        contributors += 1
                                        color = self.colors[self.patterns[t][dx + dy * self.N]]
                                        if isinstance(color, (int, float)):
                                            r, g, b = int(color)
                                        else:
                                            r += int(color[0])
                                            g += int(color[1])
                                            b += int(color[2])

                    if contributors > 0:
                        bitmap_data[x + y *self.image_width] = (int(r / contributors), int(g / contributors), int(b / contributors))
                    else:
                        print("No contributors")
                        bitmap_data[x + y * self.image_width] = (int(r), int(g), int(b))
        
        result.putdata(bitmap_data)
        return result
    
    def Clear(self):
        super(OverlappingModel, self).Clear()
        if(self.ground != 0):
            for x in range(0, self.image_width):
                for t in range(0, self.T):
                    if t != self.ground:
                        self.wave[x][self.image_height - 1][t] = False
                    self.changes[x][self.image_height - 1] = True

                    for y in range(0, self.image_height - 1):
                        self.wave[x][y][self.ground] = False
                        self.changes[x][y] = True
            
            while self.Propagate():
                pass


### some global definitions that will be needed by each class ###

def StuffRandom(source_array, random_value):
    a_sum = sum(source_array)

    if 0 == a_sum:
        for j in range(0, len(source_array)):
            source_array[j] = 1
        a_sum = sum(source_array)
    for j in range(0, len(source_array)):
        source_array[j] /= a_sum
    
    # LETS MAKE SOME NOISE
    i = 0
    x = 0
    while (i < len(source_array)):
        x += source_array[i]
        if random_value <= x:
            return i 
        i += 1
    return 0

def StuffPower(a, n):
    product = 1
    for i in range(0, n):
        product *= a
    return product

def string2bool(strn):
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["true"]

class Program:
    def __init__(self):
        pass

    def Main(self):
        self.random = random.Random()
        xdoc = ET.ElementTree(file="samples.xml")
        counter = 1
        for xnode in xdoc.getroot():
            if("#comment" == xnode.tag):
                continue
            a_model = None

            name = xnode.get('name', "NAME")
            global hackstring
            hackstring = name

            print("< {0} ".format(name), end='')
            a_model = OverlappingModel(int(xnode.get('width')), int(xnode.get('height')), xnode.get('name'), int(xnode.get('N')), int(xnode.get('symmetry')), int(xnode.get('ground')), string2bool(xnode.get('HighScore')))
            
            for i in range(0, int(xnode.get("screenshots", 2))):
                for k in range(0, 10):
                    print("> ", end='')
                    seed = self.random.random()
                    finished = a_model.Run(seed, int(xnode.get("limit", 0)))
                    if finished:
                        print("DONE")
                        a_model.Graphics().save("{0}_{1}_{2}_{3}.png".format(counter, name, i, uuid.uuid4()), format="PNG")
                        break
                    else:
                        print("Something went wrong...")
            counter += 1


if __name__ == "__main__":
    start = time.time()

    program = Program()
    program.Main()

    end = time.time()
    print(end-start)
