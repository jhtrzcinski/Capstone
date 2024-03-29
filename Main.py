import time
import random
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import uuid
import ast
import OverlappingModel as OM
import score

## need to save names of the winners somehow to pass between iterations

class GAN:
    """
    generative adversarial network object:
    -opens initial sample
    -generates a child image from the sample
    -save child image
    -write child image to xml
    -child image and parent image have scores assessed
    -write scores to xml
    -winner becomes new sample
    -loop
    """
    def __init__(self):
        pass
    
    def generate_child(self, xnode):
        self.random = random.Random()
        name = xnode.get('name', "NAME")
        a_model = OM.OverlappingModel(int(xnode.get('width')), int(xnode.get('height')), xnode.get('name'), int(xnode.get('N')), int(xnode.get('symmetry')), int(xnode.get('ground')), string2bool(xnode.get('HighScore')))
        seed = self.random.random()
        finished = a_model.Run(seed, 0)
        if finished:
            print("Child Image Generated Successfully")
            
            save_string = str("{0}_child_{1}".format(name, uuid.uuid4()))
            a_model.Graphics().save('Samples//' + save_string + '.png', format="PNG")
            return save_string
        else:
            print('Child image generation failed...')
            return 
    
    def Main(self):
        t_end = time.time() + 60 * 60 * 7
        while time.time() < t_end:
            xdoc = ET.parse("samples.xml")
            root = xdoc.getroot()
            parent = root.find(".//Overlapping[@HighScore='True']")

            if parent is None:
                print("No elements found with \'HighScore = True\' found.")
                break
            print("Parent found:", parent.get('name'))

            child = self.generate_child(self, parent)
            
            child_attrs = {
                'name': str(child),
                'width': parent.get('width'),
                'height': parent.get('height'),
                'N': parent.get('N'),
                'symmetry': parent.get('symmetry'),
                'ground': parent.get('ground'),
                'HighScore': 'False'
            }
            child_xml = ET.Element('Overlapping', child_attrs)

            parent_text = parent.text.strip('[]').strip()
            parent_score = np.array([float(parent_text.split()[0]), float(parent_text.split()[1])])
            #print(parent_score)
            child_scoring = score.Cost(int(child_xml.get('width')), int(child_xml.get('height')), child_xml.get('name'), int(child_xml.get('N')), int(child_xml.get('symmetry')), int(child_xml.get('ground')), string2bool(child_xml.get('HighScore')))
            child_score = child_scoring.run()
            child_xml.text = str(child_score)
            child_xml.tail = '\n    '

            if (parent_score[0] < child_score[0]) and (parent_score[1] > child_score[1]):
                #print('switching true and false')
                parent.set('HighScore', 'False')
                child_xml.set('HighScore', 'True')
            
            root.append(child_xml)
            xdoc.write("samples.xml")

        return
        

def string2bool(strn):
    if isinstance(strn, bool):
        return strn
    return strn.lower() in ["True"]

if __name__ == '__main__':
    #start = time.time()
    xdoc = ET.ElementTree(file='samples.xml')
    for xnode in xdoc.getroot():
        name = str(xnode.get('name'))
        if name == 'Capstone':
            sample_scoring = score.Cost(int(xnode.get('width')), int(xnode.get('height')), xnode.get('name'), int(xnode.get('N')), int(xnode.get('symmetry')), int(xnode.get('ground')), string2bool(xnode.get('HighScore')))
            sample_score = sample_scoring.run()
            xnode.text = str(sample_score)
            xnode.set('HighScore', "True")
            print('set parent to true')
            xdoc.write('samples.xml')
    gan = GAN
    gan.Main(gan)
    #end = time.time()
    #print(f"Time Elapsed: {0}".format(end-start))
