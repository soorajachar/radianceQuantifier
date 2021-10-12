#! /usr/bin/env python3
from inVivoRadianceProcessing import fullInVivoImageProcessingPipeline,selectMatrices,plotMouseImages
import argparse,sys

#Temporary; will add GUI for plotting both radiance statistics and raw mouse images soon
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sampleNameFilePath", help="path to sample name file; must be in excel or csv format")
    parser.add_argument("imageFilePath", help="path to directory containing radiance images")
    args = parser.parse_args()
    if '.csv' in args.sampleNameFilePath:
        sampleNameFile = pd.read_csv(sampleNameFilePath)
    elif '.xlsx' in args.sampleNameFilePath:
        sampleNameFile = pd.read_excel(sampleNameFilePath)
    else:
        print('Error; sample name file must be in excel or csv format. Quitting...')
        sys.exit(0)
    radianceStatisticDf = fullInvivoImageProcessingPipeline(sampleNameFile,args.imageFilePath,save_pixel_df=True)
    print(radianceStatisticDf)
