import pickle
import math
import sys
import heapq
import os
import torchvision
from scipy.stats import pearsonr  
from numpy import dot
import shutil
from numpy.linalg import norm 
from matplotlib import pyplot as plt

if len(sys.argv) < 2:
    print("Image id is needed for computing similar images. Exiting!")
    sys.exit()

with open('imageDescriptors.pickle', 'rb') as handle:
    descriptor_data = pickle.load(handle)

def Intersection_Similarity(A,B):
    min_val = 0
    max_val = 0
    for a,b in zip(A,B):
        min_val += min(a,b)
        max_val += max(a,b)
    return min_val/max_val

def Cosine_Similarity(A,B):
    return dot(A, B) / (norm(A)*norm(B))

def Pearson_Corr(A,B):
    return pearsonr(A,B)

def Manhattan_Distance(A,B):
    md = 0
    for v1, v2 in zip(A,B):
        md += abs(v1-v2)
    return md

def Euclidean_distance(A,B):
    ed = 0
    for i in range(len(A)):
        ed += (A[i] - B[i])**2
    ed = math.sqrt(ed)
    return ed
results = {}

for target in range(1, len(sys.argv)):
    
    targetImageId = int(sys.argv[target])
    print(targetImageId)
    hogDistanceValues = {}
    resnet50_fc_values = {}
    resnet50_avg_values = {}
    resnet50_layer3_values = {}
    colorMoments_values = {}

    for imageId in descriptor_data.keys():
            if imageId != targetImageId:
                hogDistanceValues[imageId] = Intersection_Similarity(descriptor_data[int(imageId)]['HOG'], descriptor_data[int(targetImageId)]['HOG'])
                resnet50_fc_values[imageId] = Cosine_Similarity(descriptor_data[int(imageId)]['Resnet50_FC'], descriptor_data[int(targetImageId)]['Resnet50_FC'])
                resnet50_avg_values[imageId] = Pearson_Corr(descriptor_data[int(imageId)]['Resnet50_Avg'], descriptor_data[int(targetImageId)]['Resnet50_Avg'])
                resnet50_layer3_values[imageId] = Pearson_Corr(descriptor_data[int(imageId)]['Resnet50_Layer3'], descriptor_data[int(targetImageId)]['Resnet50_Layer3'])
                colorMoments_values[imageId] = Pearson_Corr(descriptor_data[int(imageId)]['ColorMoments'], descriptor_data[int(targetImageId)]['ColorMoments'])
    results[targetImageId] = {}
    results[targetImageId]['HOG'] = heapq.nlargest(10, hogDistanceValues, key=hogDistanceValues.get)
    results[targetImageId]['ColorMoments'] = heapq.nlargest(10, colorMoments_values, key=colorMoments_values.get)
    results[targetImageId]['Resnet_AvgPool'] = heapq.nlargest(10, resnet50_fc_values, key=resnet50_fc_values.get)
    results[targetImageId]['Resnet_FC'] = heapq.nlargest(10, resnet50_avg_values, key=resnet50_avg_values.get)
    results[targetImageId]['Resnet_Layer3'] = heapq.nlargest(10, resnet50_layer3_values, key=resnet50_layer3_values.get)

def generateOutpus(results):
    
    dataset = torchvision.datasets.Caltech101(root='.', download=True)
    
    if not os.path.exists('./Outputs'):
        os.makedirs('./Outputs')
    else:
        shutil.rmtree('./Outputs')
        os.makedirs('./Outputs')
    for imageID in results:
        count = 11
        os.makedirs('./Outputs/'+'/'+str(imageID))
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(6, 10, 1)
        plt.imshow(dataset[int(imageID)][0])
        plt.axis('off')
        plt.title("Query Image: "+str(imageID))
        for feature in results[imageID]:
            image_list = []
            os.makedirs('./Outputs/'+'/'+str(imageID)+'/'+feature)
            for image in results[imageID][feature]:
                img, label = dataset[int(image)]
                img.save('./Outputs/'+'/'+str(imageID)+'/'+feature+'/'+str(image)+'.jpg')
                fig.add_subplot(6, 10, count)
                plt.imshow(img)
                plt.xlabel('ID: '+str(image))
                plt.xticks([])
                plt.yticks([])
                
                if count%10==1:
                    plt.title(feature+" feature")
                image_list.append(img)
                count += 1
        fig.savefig(str(imageID)+'.jpg')

generateOutpus(results)