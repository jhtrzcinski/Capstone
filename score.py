import numpy as np
import OverlappingModel
import time
from PIL import Image
import xml.etree.ElementTree as ET
from PIL import Image

class Cost:
    def __init__(self, width, height, name, N_value, symmetry_value, ground, highscore):
        """
        a little bit of image preprocessing so that the program can more easily understand the images
        """
        self.image_height = height
        self.image_width = width
        self.score = np.array([0,0])
        self.bitmap = Image.open("Samples/{0}.png".format(name))
        self.SMX = self.bitmap.size[0]
        self.SMY = self.bitmap.size[1]

        self.sample = [[0 for _ in range(self.SMY)] for _ in range(self.SMX)]
        self.colors = []

        # simplify the image into an array of color strings
        for x in range(self.SMX):
            for y in range(self.SMY):
                a_color = self.bitmap.getpixel((x, y))
                a_color = self.rgb2strn(a_color)
                color_exists = [c for c in self.colors if c == a_color]
                if len(color_exists) < 1:
                    self.colors.append(a_color)
                #sample_result = [i for i,v in enumerate(self.colors) if v == a_color]
                self.sample[x][y] = a_color

        self.color_count = len(self.colors)

        # since we find out how many colors we have here, might as well apply the punishment for missing colors
        if self.color_count < 9:
            # subtract a million for every color short it is
            self.score[0] -= 1e6*(9-self.color_count)
        #print(self.sample[0])

        return

    def rgb2strn(self, a_color):
        rgb = []
        color = "blank"
        for val in a_color:
            rgb.append(val)

        if rgb == [195, 195, 195]:
            color = "gray"
        if rgb == [255, 174, 201]:
            color = "pink"
        if rgb == [255, 127, 39]:
            color = "orange"
        if rgb == [237, 28, 36]:
            color = "red"
        if rgb == [163, 73, 164]:
            color = "purple"
        if rgb == [34, 177, 76]:
            color = "green"
        if rgb == [63, 72, 204]:
            color = "blue"
        if rgb == [0, 162, 232]:
            color = "light blue"
        if rgb == [255, 242, 0]:
            color = "yellow"

        return color
        

    def color_center(self):
        """
        finds the center of each color so that the distances can be calculated
        """
        color_centers = []

        for x in range(self.SMX):
            for y in range(self.SMY):
                color = self.sample[x][y]
                if color in color_centers:
                    index_color = color_centers.index(color)
                    count, x_sum, y_sum = color_centers[index_color]
                    count += 1
                    x_sum += x
                    y_sum += y
                    color_centers[index_color] = (count, x_sum, y_sum)
                else:
                    color_centers.append((1, x, y))

        for color in color_centers:
            index_color = color_centers.index(color)
            count, x_sum, y_sum = color_centers[index_color]
            center_x = x_sum / count
            center_y = y_sum / count
            color_centers[index_color] = (center_x, center_y)

        return color_centers

    def color_separation(self, color1, color2, color_centers):
        separation = 0
        if (str(color1) in self.colors) and (str(color2) in self.colors):
            if (str(color1) in self.colors):
                color1 = color_centers[self.colors.index(str(color1))]
                color2 = color_centers[self.colors.index(str(color2))]
                separation += np.sqrt((color1[0]-color2[0])**2 + (color1[1]-color2[1])**2)
            else: separation += 200
        else: separation += 100
        return separation
        
    def calculate_separation(self):
        """
        calculates the seperation between components' centers
        """
        ### calculate: (Red-Orange), (Orange-yellow), (orange-blue), (orange-lblue), (orange-purple), (orange-pink), (yellow-green), (blue-lblue), (yellow-blue), (yellow-purple)
        color_centers = self.color_center() # returns center coordinate in position corresponding to color index in colors
        separation = 0

        separation += self.color_separation('red', 'orange', color_centers)
        separation += self.color_separation('yellow', 'orange', color_centers)
        separation += self.color_separation('blue', 'orange', color_centers)
        separation += self.color_separation('light blue', 'orange', color_centers)
        separation += self.color_separation('purple', 'orange', color_centers)
        separation += self.color_separation('pink', 'orange', color_centers)
        separation += self.color_separation('green', 'yellow', color_centers)
        separation += self.color_separation('blue', 'yellow', color_centers)
        separation += self.color_separation('purple', 'yellow', color_centers)
        separation += self.color_separation('blue', 'light blue', color_centers)

        if separation == 0:
            separation += 1e7

        return separation
        
    def check_together(self):
        """
        check if the color components are together by calculating the sum of sides that are like,
        higher is better
        """
        width, height = self.image_width, self.image_height

        pixels = self.sample
        
        together_array = np.zeros([width, height])

        for x in range(width):
            for y in range(height):
                color = pixels[x][y]
                if color == "gray":
                    pass # we dont care if the gray bits arent adjacent, they are filler
                sides = 4
                if x == 0 or x == width:
                    sides += -1
                if y == 0 or y == height:
                    sides += -1
                if x > 0:
                    left_color = pixels[x-1][y]
                    if left_color != color:
                        sides += -1
                if x < width-1:
                    right_color = pixels[x+1][y]
                    if right_color != color:
                        sides += -1
                if y > 0:
                    up_color = pixels[x][y-1]
                    if up_color != color:
                        sides += -1
                if y < height-1:
                    down_color = pixels[x][y+1]
                    if down_color != color:
                        sides += -1
                together_array[x][y] = sides
                
        ### might want to calculate if the colors are actually together, if not, punish
        
        return together_array.sum()

    def run(self):
        """
        This will run the functions to calculate seperation, and score its color usage
        """
        self.score[0] += self.check_together()
        self.score[1] += self.calculate_separation()

        return self.score

def string2bool(strn):
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["true"]

if __name__ == '__main__':
    start = time.time()
 
    xdoc = ET.ElementTree(file='samples.xml')
    for xnode in xdoc.getroot():
        a_model = None

        name = xnode.get("name", "NAME")
        global hackstring
        hackstring = name
        item = Cost(int(xnode.get('width'), int(xnode.get('height')), xnode.get('name'), int(xnode.get('N')), int(xnode.get('symmetry')), int(xnode.get('ground')), string2bool(xnode.get('HighScore'))))
        print(name, item.run())

        #if (string2bool(xnode.get('samplebool', "bool")) == True) and (string2bool(xnode.get('generatedbool', "bool")) == False):
        #    score1 = Cost(int(xnode.get('width', 48)), int(xnode.get('height', 48)), xnode.get('name', "NAME"), int(xnode.get('N', 2)), int(xnode.get('symmetry', 8)), int(xnode.get('ground',0)), string2bool(xnode.get('generatedBool',"False")), string2bool(xnode.get('sampleBool',"True")), np.array(xnode.get('scorearr', [0, 0])))
        #if (string2bool(xnode.get('generatedbool', "bool"))== True) and (string2bool(xnode.get('samplebool', "bool")) == False):
        #    score2 = Cost(int(xnode.get('width', 48)), int(xnode.get('height', 48)), xnode.get('name', "NAME"), int(xnode.get('N', 2)), int(xnode.get('symmetry', 8)), int(xnode.get('ground',0)), string2bool(xnode.get('generatedBool',"True")), string2bool(xnode.get('sampleBool',"True")), np.array(xnode.get('scorearr', [0, 0])))
    
    #score_sample = score1.run(sample)
    #score_generated = score2.run(generated)
    end = time.time()
    print(end-start)