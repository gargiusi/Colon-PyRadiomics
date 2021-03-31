import pydicom as dicom
from pydicom import dcmread,dcmwrite
import pydi
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
#from skimage.viewer import ImageViewer
from skimage.io import imread
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from pydicom.dataset import Dataset,FileDataset
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xlwt
from radiomics import featureextractor, getFeatureClasses
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape,gldm,ngtdm
import SimpleITK as sitk
import six
import os
import tkinter
from tkinter import filedialog,messagebox
import pywt
import pywt.data
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from dipy.segment.mask import median_otsu
import ctypes
import pymsgbox
from glob import glob
import nrrd
import dicom2nifti
import dicom2nifti.settings as settings
import gc
import cv2
import sys
import skimage.transform as skTrans
from scipy import ndimage as ndi
import scipy.misc
from glob import glob
import warnings
warnings.simplefilter('ignore')

workbook=xlwt.Workbook()

try:

        root = tkinter.Tk()
        root.withdraw()
        currdir = os.getcwd()

        pathname = filedialog.askdirectory(parent=root, initialdir=currdir, title='Please select the Archive of all the Patients')

        if os.path.isdir(pathname) != True:
            exit()

        def maskType():
            MaskType = pymsgbox.prompt('Which mask you want to use: Enter 1 for Label mask; 2 for Cropped mask; 3 for both')
            MaskType = int(MaskType)
            if MaskType != 1 and MaskType != 2 and MaskType != 3:
                pymsgbox.alert('Please enter a number of this range [1,2,3]', 'Warning')
                maskType()
            return MaskType

        def imageType():
            ImageType = pymsgbox.prompt('Enter 1 for Original Image Extraction; 2 for Wavelet based Image Extraction; 3 for both')
            ImageType = int(ImageType)
            if ImageType != 1 and ImageType != 2 and ImageType != 3:
                pymsgbox.alert('Please enter a number of this range [1,2,3]', 'Warning')
                imageType()
            return ImageType

        MaskType=maskType()

        ImageType=imageType()

        if ImageType==2 or ImageType==3:
            applyWavelet = True
        else:
            applyWavelet = False
        if ImageType == 1 or ImageType==3:
            applyOriginalImage = True
        else:
            applyOriginalImage = False

        if applyWavelet:
            ExtractPNG = messagebox.askquestion("Extract Image", "You want to save all slices as PNG format?")
            if ExtractPNG == 'yes':
                ExtractPNG = True
            else:
                ExtractPNG = False
        else:
            ExtractPNG = False

        P=0
        w=0
        List=os.listdir(pathname)


        for C in List:
          percorsoPaziente= pathname+'/'+str(C)
          Esistente = False
          for tt in os.listdir(percorsoPaziente):
            if tt=='NIFTI_FILE':
              Esistente=True
          if Esistente==False:
              os.mkdir(percorsoPaziente+'/NIFTI_FILE')
          if os.path.isdir(percorsoPaziente) == True and C!='__MACOSX':

            Esistente1=False
            P=P+1

            print("Patient N."+str(P))
            for tt in os.listdir(percorsoPaziente+'/NIFTI_FILE'):
                Esistente1 = True
            if Esistente1 == False:
                if P==1:
                    #settings.disable_validate_slice_increment()
                    #settings.disable_validate_orthogonal()
                    settings.enable_resampling()
                    settings.set_resample_spline_interpolation_order(1)
                    settings.set_resample_padding(-1000)
                dicom2nifti.convert_dir.convert_directory(percorsoPaziente+"\\IMA",percorsoPaziente+"\\NIFTI_FILE",compression=True,reorient=True)

            testcase=str(C)
            DataDir=percorsoPaziente+'/ROI'
            FileNIFTI=os.listdir(percorsoPaziente+'/NIFTI_FILE')
            for LP in FileNIFTI:
                filenitfi=percorsoPaziente+'/NIFTI_FILE/'+LP
                ImagePath=str(filenitfi)

            MaskFile=np.array((1,2),dtype=str)
            definizioni=np.array((1,2),dtype=str)
            Name=np.array((1,2),dtype=str)
            for tp in os.listdir(DataDir):
                roi=str(tp)
                if (roi.find('label')!= -1):
                    roiLabelImage =roi
                if (roi.find('cropped')!= -1):
                    roiCroppedImage =roi

            MaskFile=[DataDir+'/'+roiLabelImage,DataDir + '/' + roiCroppedImage]
            definizioni=[DataDir+'/'+testcase+'_roi_label.',DataDir + '/' + testcase + '_roi_cropped.']
            m=0
            if MaskType ==1:
                INIT=0
            if MaskType==2:
                INIT=1
            if MaskType ==3:
                INIT=0
                MaskType=2
            Name=['Label mask','Crop mask']
            for md in range(INIT,MaskType):
                LabelPath=MaskFile[md]
                maskfileNifti=definizioni[m]
                m=m+1
                n = 1
                #Settings for Features Extraction

                settings={'binWidth': 25,
                        'interpolator': sitk.sitkBSpline,
                        'resampledPixelSpacing': [1,1,2],
                        'padDistance': 5,
                        'normalize': False,
                        'normalizeScale': 1,
                        'correctMask': True,
                        'symmetricalGLCM': True,
                        'minimumROIDimensions': 2,
                        'label': 1,
                        'additionalInfo': True,
                        'force2Ddimension': 0}
                image = sitk.ReadImage(ImagePath)
                mask = sitk.ReadImage(LabelPath)



                if ExtractPNG:
                    titles = ['Approximation (LL)', ' Horizontal detail (LH)',
                              'Vertical detail (HL)', 'Diagonal detail (HH)']
                    test_image = nib.load(ImagePath).get_fdata()
                    p=0
                    Esistente3=False
                    for tt in os.listdir(percorsoPaziente):
                        if tt == 'Image PNG':
                            Esistente3 = True
                    if Esistente3 == False:
                        os.mkdir(percorsoPaziente + '/Image PNG')
                        while p!= test_image.shape[2]:
                            slice = test_image[:,:,p]
                            im=Image.fromarray(slice)
                            coeffs2 = pywt.dwt2(im, 'haar',mode='symmetric')
                            LL, (LH, HL, HH) = coeffs2
                            fig = plt.figure(figsize=(12, 3))
                            for i, a in enumerate([LL, LH, HL, HH]):
                                ax = fig.add_subplot(1, 5, i + 1+1)
                                ax.imshow(a, interpolation="nearest",cmap=plt.cm.gray)
                                ax.set_title(titles[i], fontsize=10)
                                ax.set_xticks([])
                                ax.set_yticks([])

                            ax = fig.add_subplot(1, 5, 1)
                            ax.imshow(slice, interpolation="nearest", cmap=plt.cm.gray)
                            ax.set_title('Original', fontsize=10)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            fig.tight_layout()
                            #plt.show()
                            fig.savefig(percorsoPaziente + '/Image PNG/'+testcase+str(p)+'.png')
                            p=p+1



                interpolator = settings.get('interpolator')
                resampledPixelSpacing = settings.get('resampledPixelSpacing')
                if interpolator is not None and resampledPixelSpacing is not None:
                    image, mask = imageoperations.resampleImage(image, mask, **settings)
                bb, correctedMask = imageoperations.checkMask(image, mask,**settings)
                if correctedMask is not None:
                    mask = correctedMask
                if applyWavelet==False:
                    image, mask = imageoperations.cropToTumorMask(image, mask, bb)

                if applyOriginalImage:
                        # Show Shape features
                        shapeFeatures = shape.RadiomicsShape(image, mask, **settings)
                        shapeFeatures.enableAllFeatures()

                        print('Will calculate the following Shape features: ')
                        for f in shapeFeatures.enabledFeatures.keys():
                          print('  ', f)
                          print(getattr(shapeFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating Shape features...')
                        results = shapeFeatures.execute()
                        print('done')

                        print('Calculated Shape features: ')
                        for (key, val) in six.iteritems(results):
                          print('  ', key, ':', val)

                        if m==1:
                            if P == 1:
                              sheet1 = workbook.add_sheet('Original Image-'+Name[INIT])
                              sheet1.write(0, 0, 'Patient_ID')
                            sheet1.write(P, 0, str(testcase))
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            if P == 1:
                              sheet1_2 = workbook.add_sheet('Original Image-'+Name[INIT+1])
                              sheet1_2.write(0, 0, 'Patient_ID')
                            sheet1_2.write(P, 0, str(testcase))
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del shapeFeatures

                        # Show the first order feature calculations

                        firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **settings)
                        firstOrderFeatures.enableAllFeatures()

                        print('Will calculate the following first order features: ')
                        for f in firstOrderFeatures.enabledFeatures.keys():
                          print('  ', f)
                          print(getattr(firstOrderFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating first order features...')
                        results = firstOrderFeatures.execute()
                        print('done')

                        print('Calculated first order features: ')
                        for (key, val) in six.iteritems(results):
                          print('  ', key, ':', val)

                        if m==1:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del firstOrderFeatures

                        # Show GLCM features

                        glcmFeatures = glcm.RadiomicsGLCM(image,mask, **settings)
                        glcmFeatures.enableAllFeatures()
                        print('Will calculate the following GLCM features: ')
                        for f in glcmFeatures.enabledFeatures.keys():
                          print('  ', f)
                          print(getattr(glcmFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating GLCM features...')
                        results = glcmFeatures.execute()
                        print('done')

                        print('Calculated GLCM features: ')
                        for (key, val) in six.iteritems(results):
                          print('  ', key, ':', val)

                        if m==1:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del glcmFeatures


                        # Show GLDM features

                        gldmFeatures =gldm.RadiomicsGLDM(image, mask, **settings)
                        gldmFeatures.enableAllFeatures()
                        print('Will calculate the following GLDM features: ')
                        for f in gldmFeatures.enabledFeatures.keys():
                            print('  ', f)
                            print(getattr(gldmFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating GLDM features...')
                        results = gldmFeatures.execute()
                        print('done')

                        print('Calculated GLDM features: ')
                        for (key, val) in six.iteritems(results):
                            print('  ', key, ':', val)

                        if m==1:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del gldmFeatures


                        # Show GLRLM features
                        #
                        glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
                        glrlmFeatures.enableAllFeatures()

                        print('Will calculate the following GLRLM features: ')
                        for f in glrlmFeatures.enabledFeatures.keys():
                          print('  ', f)
                          print(getattr(glrlmFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating GLRLM features...')
                        results = glrlmFeatures.execute()
                        print('done')

                        print('Calculated GLRLM features: ')
                        for (key, val) in six.iteritems(results):
                          print('  ', key, ':', val)

                        if m==1:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del glrlmFeatures

                        # Show GLSZM features
                        #
                        glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
                        glszmFeatures.enableAllFeatures()

                        print('Will calculate the following GLSZM features: ')
                        for f in glszmFeatures.enabledFeatures.keys():
                          print('  ', f)
                          print(getattr(glszmFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating GLSZM features...')
                        results = glszmFeatures.execute()
                        print('done')

                        print('Calculated GLSZM features: ')
                        for (key, val) in six.iteritems(results):
                          print('  ', key, ':', val)

                        if m==1:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del glszmFeatures


                        # Show NGTDM features
                        #
                        ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
                        ngtdmFeatures.enableAllFeatures()

                        print('Will calculate the following NGTDM features: ')
                        for f in ngtdmFeatures.enabledFeatures.keys():
                          print('  ', f)
                          print(getattr(ngtdmFeatures, 'get%sFeatureValue' % f).__doc__)

                        print('Calculating NGTDM features...')
                        results = ngtdmFeatures.execute()
                        print('done')

                        print('Calculated NGTDM features: ')
                        for (key, val) in six.iteritems(results):
                          print('  ', key, ':', val)

                        if m==1:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1.write(0, n, str(key))
                                sheet1.write(P, n, str(val))
                                n = n + 1
                        if m==2:
                            for key, val in results.items():
                                if P == 1:
                                  sheet1_2.write(0, n, str(key))
                                sheet1_2.write(P, n, str(val))
                                n = n + 1
                        del results
                        del ngtdmFeatures



                if applyWavelet:

                    for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image,
                                                                                                              mask):
                        k = 2
                        w = w + 1
                        waveletShapeFeaturs = shape.RadiomicsShape(decompositionImage, mask, **settings)
                        waveletShapeFeaturs.enableAllFeatures()
                        results = waveletShapeFeaturs.execute()


                        print('Calculated ShapeFeaturs features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            if w == 1:
                                sheet2 = workbook.add_sheet('Wavelet based image-' + Name[INIT])
                                sheet2.write(0, 0, 'Patient_ID')
                                sheet2.write(0, 1, 'Decomposition Name')

                            sheet2.write(w, 0, str(testcase))
                            sheet2.write(w, 1, str(decompositionName))
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            if w == 1:
                                sheet2_2 = workbook.add_sheet('Wavelet based image-' + Name[INIT + 1])
                                sheet2_2.write(0, 0, 'Patient_ID')
                                sheet2_2.write(0, 1, 'Decomposition Name')

                            sheet2_2.write(w, 0, str(testcase))
                            sheet2_2.write(w, 1, str(decompositionName))
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletShapeFeaturs

                        waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **settings)
                        waveletFirstOrderFeaturs.enableAllFeatures()
                        results = waveletFirstOrderFeaturs.execute()
                        print('Calculated firstorder features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletFirstOrderFeaturs

                        waveletglcmFeatures = glcm.RadiomicsGLCM(decompositionImage, mask, **settings)
                        waveletglcmFeatures.enableAllFeatures()
                        results = waveletglcmFeatures.execute()
                        print('Calculated glcmFeatures features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletglcmFeatures

                        waveletgldmFeatures = gldm.RadiomicsGLDM(decompositionImage, mask, **settings)
                        waveletgldmFeatures.enableAllFeatures()
                        results = waveletgldmFeatures.execute()
                        print('Calculated gldmFeatures features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletgldmFeatures

                        waveletglrlmFeatures = glrlm.RadiomicsGLRLM(decompositionImage, mask, **settings)
                        waveletglrlmFeatures.enableAllFeatures()
                        results = waveletglrlmFeatures.execute()
                        print('Calculated glrlmFeatures features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletglrlmFeatures

                        waveletglszmFeatures = glszm.RadiomicsGLSZM(decompositionImage, mask, **settings)
                        waveletglszmFeatures.enableAllFeatures()
                        results = waveletglszmFeatures.execute()
                        print('Calculated glszmFeatures features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletglszmFeatures

                        waveletngtdmFeatures = ngtdm.RadiomicsNGTDM(decompositionImage, mask, **settings)
                        waveletngtdmFeatures.enableAllFeatures()
                        results = waveletngtdmFeatures.execute()
                        print('Calculated ngtdmFeatures features with wavelet ', decompositionName)
                        for (key, val) in six.iteritems(results):
                            waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                            print('  ', waveletFeatureName, ':', val)

                        if m == 1:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2.write(0, k, str(key))
                                sheet2.write(w, k, str(val))
                                k = k + 1
                        if m == 2:
                            for key, val in results.items():
                                if w == 1:
                                    sheet2_2.write(0, k, str(key))
                                sheet2_2.write(w, k, str(val))
                                k = k + 1
                        del results
                        del waveletngtdmFeatures
                        gc.collect()


        print('Finish')
        def SaveAs():
            pathname1 = filedialog.askdirectory(parent=root, initialdir=currdir, title='SaveAs: Select a directory')
            if os.path.isdir(pathname1) != True:
                SaveAs()
            return pathname1

        pathname1=SaveAs()
        workbook.save(pathname1+'/Results.csv')

except:
    def SaveAs():
        pathname1 = filedialog.askdirectory(parent=root, initialdir=currdir, title='SaveAs: Select a directory')
        if os.path.isdir(pathname1) != True:
            SaveAs()
        return pathname1


    print('Il paziente:'+testcase+' ha dato un errore')
    pathname1 = SaveAs()
    workbook.save(pathname1 + '/Results.csv')

