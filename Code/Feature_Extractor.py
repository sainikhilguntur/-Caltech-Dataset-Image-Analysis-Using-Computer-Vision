import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
import pickle
from statistics import mean
import tqdm
import sys

print("*********Welcome to Phase 1 project of CSE 515*********")
if len(sys.argv) < 2:
    print(" For task 1, provide an image ID for calculating feature models for a given imageID. \n For task2 give 'ALL' as argument to compute feature models for all 8677 images!")
    sys.exit()

# Loading the Caltech dataset using toechvision datasets. This will download the dataset only when it is not available locally. 
dataset = torchvision.datasets.Caltech101(root='.', download=True)    
# Setting the data loader with batch size.                                    
downdata_loader = torch.utils.data.DataLoader(dataset,
batch_size=4,
shuffle=True,
num_workers=8)

# Color Moments Calculation
def color_moments(image):
    # Converting the image into array of pixels
    image_array = np.array(image)
    color_m_list = []
    def compute_moments(image_subarray):
        
        # Function to calculate skewness according to the formula given in lecture slides. 
        def skewness(v, pixels):
            # Calculate average value of the given block and for the given channel
            average_val = mean(v)
            temp = 0
            for val in v:
                temp += (val-average_val)*(val-average_val)*(val-average_val)
            # Compute skewness and return the same
            skew_value = math.pow((temp/len(v)), 1/3)
            return skew_value

        moments = []
        R_channel_list = []
        G_channel_list = []
        B_channel_list = []
        # For the given block, going over each pixel and creating individual lists for each channel.
        for i in range(10):
            for j in range(30):
                R_channel_list.append(image_subarray[i][j][0])
                G_channel_list.append(image_subarray[i][j][1])
                B_channel_list.append(image_subarray[i][j][2])
        # Calculating moments for each channel list obtained above.
        # 1st moment
        moments.append(mean(R_channel_list))
        moments.append(mean(G_channel_list))
        moments.append(mean(B_channel_list))
        # 2nd moment
        moments.append(np.std(R_channel_list))
        moments.append(np.std(G_channel_list))
        moments.append(np.std(B_channel_list))  
        # 3rd moment
        moments.append(skewness(R_channel_list))
        moments.append(skewness(G_channel_list))
        moments.append(skewness(B_channel_list))
        return moments
        
    # Dividing the 300x100 image into 10x10 partitions by creating a 30x10 block and calculating colormoments for each block separately in a loop.
    for i in range(0, 100, 10):
        for j in range(0,300,30):
            color_m_list += compute_moments(image_array[i:i+10,j:j+30])
    # Returning the appended color moments
    return color_m_list

# Resnet 50 features with a hook at average pool layer
def resnet_avgpool(img, res50):
    outputs= []
    # Defining the hook to be placed at avg pool layer.
    def hook(module, input, output):
        outputs.append(output)
    # Registering the hook at avg pool layer 
    hook = res50.avgpool.register_forward_hook(hook)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Transofrming image array data to tensor for fast processing. 
    image = transform(img).unsqueeze(0)

    # Running a forward pass.
    with torch.no_grad():
        out = res50(image)
    # Removing the hook as the same model will be used by other ResNet functions.
    hook.remove()
    # Obtaining 2048 length output vector from avg pool layer.
    output_2048 = outputs[0].squeeze().numpy().tolist()
    output_1024 = []
    # Going over the 2048 length output vector and replacing each value with mean of itself and the adjacent value in the array. 
    for i in range(0, len(output_2048), 2):
        if i + 1 < len(output_2048):
            avg = (output_2048[i] + output_2048[i + 1]) / 2.0
            output_1024.append(avg)
        else:
            output_1024.append(output_2048[i])
    # returning the 1024 length array.
    return output_1024

# Resnet 50 features at fc layer
def resnet_fc(img, res50):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Transorming image array to tensor
    image = transform(img).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        out = res50(image)
    # Returing output of one forward pass which will be the output of fc layer as it is the last layer in ResNet architecture.
    return out.squeeze().numpy().tolist()

# Resnet 50 features with a hook at layer 3.
def resnet_layer3(img, res50):

    outputs= []
    # Defining hook
    def hook(module, input, output):
        outputs.append(output)
    # Registering hook for layer 3
    hook = res50.layer3.register_forward_hook(hook)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Converting image array to tensor
    image = transform(img).unsqueeze(0)
    # Forward pass
    with torch.no_grad():
        out = res50(image)
    layer3_output = outputs[0].squeeze()
    # reducing the output by computing the mean of 14x14 slice which are 1,2 dimentions in the tensor.
    reduced_output = torch.mean(layer3_output.view(1024, 14, 14), dim=(1, 2))
    # removing hook
    hook.remove()
    return reduced_output.squeeze().numpy().tolist()

# HOG Feature Discriptors Calculation
def HOG(img):

    # Converting the RGB image to grayscale
    gray_image = img.convert('L')
    image_array = np.array(gray_image)
    mag_array = []
    angle_array = []
    
    # Going over every pixel in the 300x100 to calculate horizontal and vertical gradients.
    for i in range(100):
        m_array = []
        a_array = []
        for j in range(300):
            # If the current pixel is at left or right most column edge, we will be having only one value to calculate horizontal gradient.
            if j+1 >= 300 or j-1 <= 0:
                if j-1 <= 0:
                    gx = image_array[i][j+1]
                else:
                    gx = -1*(image_array[i][j-1])
            else:
                gx = image_array[i][j+1] - image_array[i][j-1]
    
            # If the current pixel is at top or bottom edge, we will be having only one value to calculate vertical gradient.
            if i+1 >= 100 or i-1 <= 0:
                if i-1 <= 0:
                    gy = -1*(image_array[i+1][j])
                else:
                    gy = image_array[i-1][j]
            else:
                gy = image_array[i-1][j] - image_array[i+1][j]
            
            # calculating magnitude of the computed gradients.
            magnitude = math.sqrt(gx**2 + gy**2)
            m_array.append(magnitude)
            # computing inverse tangent to obtain angle.
            if gx != 0:
                angle = math.degrees(abs(math.atan(gy / gx)))
            else:
                angle = math.degrees(0.0)
            a_array.append(angle)
        # Created lists containing all the magnitudes and angles of all pixels.
        mag_array.append(m_array)
        angle_array.append(a_array)
    
    def create_bins(mag_subarray, angle_subarray):
        hog = [0,0,0,0,0,0,0,0,0]
        count = 0
        # Creating vectors for given partition of pixels
        for i in range(10):
            for j in range(30):
                try:
                    # Distributing the magnitude to lower bound bin and upper bound bin based on the current angle.
                    lower_bin = math.floor(angle_subarray[i][j] / 40)
                    if lower_bin != 8:
                        lval = mag_subarray[i][j] * (((lower_bin+1)*40 - angle_subarray[i][j]) / 40)
                        hog[lower_bin] += lval
                        hog[lower_bin+1] += angle_subarray[i][j] - lval
                    else:
                        hog[8] += angle_subarray[i][j]
                    count += 1
                except:
                    print(mag_subarray)
                    print(angle_subarray)
        return hog
    
    hog_vectors = []
    np_mag_array = np.array(mag_array)
    np_angle_array = np.array(angle_array)
    # Dividing the image into 10x10 blocks and calculating hog vectors separately for each block.
    for i in range(0, 100, 10):
        for j in range(0,300,30):
            mag = np_mag_array[i:i+10,j:j+30]
            angles = np_angle_array[i:i+10,j:j+30]
            hog_vectors += create_bins(mag, angles)
    return hog_vectors

imageDescriptorsDict = {}
# Loading the pretrainied model
res50 = models.resnet50(pretrained=True)
def execute_program(image_ID):
    img, label = dataset[image_ID]
    #image_array = np.array(img)
    # If image doesn't contain all three channels, convert image to RGB
    if not isinstance(np.array(img)[0][0],np.ndarray) or np.array(img)[0][0].size != 3:
        img = img.convert('RGB')
    try:
        print(image_ID)
        # Resize image 
        img300 = img.resize((300, 100))
        img224 = img.resize((224, 224))

        # Obtain feature vectors
        hog_data = HOG(img300)
        color_moments_data = color_moments(img300)
        resnet_ap_data = resnet_avgpool(img224, res50)
        resnet_fc_data = resnet_fc(img224, res50)
        resnet_l3_data = resnet_layer3(img224, res50)
        # Create the feature descriptors dict
        imageDescriptorsDict[image_ID] = {
            'HOG':hog_data, 
            'ColorMoments':color_moments_data, 
            'Resnet50_Avg':resnet_ap_data,
            'Resnet50_FC' :resnet_fc_data,
            'Resnet50_Layer3' :resnet_l3_data
        }
    except:
        with open('imageDescriptors_temp.pickle', 'wb') as handle:
            pickle.dump(imageDescriptorsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Main program
# Initiaitng the resnet50 pretrained model


if sys.argv[1] == 'ALL':
    print("Calculating feature models for all images. Features will be stored as imageDescriptors.pickle file at the end of this program")
    counter = 0
    for image_id in tqdm.tqdm(range(8677)):
        
        execute_program(image_id)
    # Save the dict as pickle file.
    with open('imageDescriptors.pickle', 'wb') as handle:
        pickle.dump(imageDescriptorsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
else:
    print("Calculating feature models for image id - "+sys.argv[1]+"Output will be saved in "+'image_'+sys.argv[1]+'_features.txt file')
    execute_program(int(sys.argv[1]))

    file = open('image_'+sys.argv[1]+'_features.txt','w')
    
    for key in imageDescriptorsDict:
        for feature in imageDescriptorsDict[key]:
            file.write("\nFeature - "+feature+"\n")
            for val in imageDescriptorsDict[key][feature]:
                file.write(str(val)+",")
    file.close()
