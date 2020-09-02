# -*- coding: utf-8 -*-
"""

@author: ramon, bojana
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1494802, 1494946, 1491186

def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """
    groundTruth = []
    fd = open(fileName,'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
    scores = []
    for i in range(len(description)):
        img=similarityMetric(description[i],GT[i][1],options)
        scores.append(img)   
    return sum(scores)/len(scores),scores # recorda sum(scores)/len(scores) --> vol dir mean score

def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    
    if options == None: 
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'
    if options['metric'].lower() == 'basic'.lower():
        GTintersectEst=[]
        GTintersectEst=set(Est) & set(GT)
        similarity=float(len(GTintersectEst))/float(len(Est))
        return similarity

    return 0
        
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """

    unique=[]
    meaningful_colors=[]
    arrayOfCentroids=np.copy(kmeans.centroids) 
    indexColor=np.zeros((kmeans.K,1))
    indexColor=np.argmax(arrayOfCentroids,axis=1)
    
    for index in range(indexColor.shape[0]):
        if (arrayOfCentroids[index][indexColor[index]] >= options['single_thr']):# En cas que sigui etiqueta simple
            nameColor=cn.colors[indexColor[index]]            
        else:
            uxcolors=[]
            ordenColores=np.argsort(arrayOfCentroids[index])
            uxcolors.append(cn.colors[ordenColores[-1]])
            uxcolors.append(cn.colors[ordenColores[-2]])
            uxcolors.sort()
            nameColor=uxcolors[0]+uxcolors[1]
        if(nameColor not in meaningful_colors):
               meaningful_colors.append(nameColor)
               unique.append([])           
        
        npMeanColors=np.array(meaningful_colors)
        indexAux=np.where(nameColor==npMeanColors)[0]
        unique[indexAux[0]].append(index)
            
    return meaningful_colors, unique


def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """
    
    if options['colorspace'].lower() == 'ColorNaming'.lower():  
        im=cn.ImColorNamingTSELabDescriptor(im)
        
    elif options['colorspace'].lower() == 'RGB'.lower(): 
        pass
    elif options['colorspace'].lower() == 'Lab'.lower():
        im=im.astype('float64')
        im=color.rgb2lab(im/255)
        
    kmeansAlgorithm = km.KMeans(im, options['K'], options) 
    kmeansAlgorithm.run()

    if options['colorspace'].lower() == 'RGB'.lower():
        kmeansAlgorithm.centroids=cn.ImColorNamingTSELabDescriptor(kmeansAlgorithm.centroids)
        
    elif options['colorspace'].lower() == 'Lab'.lower():
        kmeansAlgorithm.centroids=color.lab2rgb([kmeansAlgorithm.centroids])[0]*255
        kmeansAlgorithm.centroids=cn.ImColorNamingTSELabDescriptor(kmeansAlgorithm.centroids)
        
                
    colors_obt, which_obt = getLabels(kmeansAlgorithm, options)   
    return colors_obt, which_obt, kmeansAlgorithm

