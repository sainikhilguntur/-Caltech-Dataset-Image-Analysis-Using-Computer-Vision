# Caltech-Dataset-Image-Analysis-Using-Computer-Vision

- To run this project, python envorinment has to be set using the requrements.txt file provided in the Code folder. 
- Once the python environemt is set, following commands can be used to run the project. 

### This command will create a file with all the features descriptors printed in that file for the given image ID.
```% python Feature_Extractor.py 0```            # Giving 0 as image id
   
### This command will compute feature descriptors for all the images in  dataset and store them as a pickle file with name imageDescriptors.pickle
```% python Feature_Extractor.py ALL```         

### This command will compute similar images for all the query image IDs passed as command line arguments and store them in outputs folder. 
```% python Similarity_Measure.py 0 880 2500 5122 8676```        
