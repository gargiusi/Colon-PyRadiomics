import pydicom as dicom
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
#from skimage.viewer import ImageViewer
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
#from dipy.segment.mask import median_otsu
import ctypes
import pymsgbox
from glob import glob
import nrrd
import dicom2nifti
import dicom2nifti.settings as settings

workbook=xlwt.Workbook()

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

MaskType=maskType()

#waveletFilter=ctypes.windll.user32.MessageBoxW(0, "You want to extract also wavelet based features?", "Wavelet Filter", 3)
waveletFilter=messagebox.askquestion("Wavelet Filter","You want to extract also wavelet based features?")
if waveletFilter=='yes':
    applyWavelet = True
else:
    applyWavelet = False

P=0
w=0
List=os.listdir(pathname)

for C in List:
  percorsoPaziente=pathname+'/'+str(C)
  Esistente = False
  for tt in os.listdir(percorsoPaziente):
    if tt=='NIFTI_FILE':
      Esistente=True
  if Esistente==False:
      os.mkdir(percorsoPaziente+'/NIFTI_FILE')
  if os.path.isdir(percorsoPaziente) == True and C!='__MACOSX':

    #dataset = dicom.read_file(percorsoPaziente+'/IMA/'+str(C)+'0.dcm')
    #print(dataset)
    # plt.imshow(dataset.pixel_array,plt.cm.bone)
    # plt.show()
    Esistente1=False
    P=P+1

    print("Patient N."+str(P))
    #for tt in os.listdir(percorsoPaziente+'/NIFTI_FILE'):
        #Esistente1 = True
    if Esistente1 == False:
        if P==1:
            settings.disable_validate_slice_increment()
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
    MaskFile=[DataDir+'/'+testcase+'_roi_T_label.nrrd',DataDir + '/' + testcase + '_roi_T_cropped.nrrd']
    definizioni=[DataDir+'/'+testcase+'_roi_T_label.',DataDir + '/' + testcase + '_roi_T_cropped.']
    m=0
    if MaskType ==1:
        INIT=0
    if MaskType==2:
        INIT=1
    if MaskType ==3:
        INIT=0
        MaskType=2
    Name=['Label mask','Cropped mask']
    for md in range(INIT,MaskType):
        LabelPath=MaskFile[md]
        maskfileNifti=definizioni[m]
        m=m+1
        n = 1
        #Settings for Features Extraction

        settings={'binWidth': 25,
                'interpolator': 'sitkBSpline',
                'correctMask': True}
        image = sitk.ReadImage(ImagePath)
        mask = sitk.ReadImage(LabelPath)

        img = nib.load(ImagePath)
        data = img.get_fdata()
        Size=data.shape

        # Save As NRRD mask file To NIFTI mask file in the same directory

        _nrrd = nrrd.read(LabelPath)
        data_nrrd = _nrrd[0]
        header = _nrrd[1]
        img_nrrd = nib.Nifti1Image(data_nrrd, np.eye(4))
        nib.save(img_nrrd, maskfileNifti+'nii.gz')
        mask = sitk.ReadImage(maskfileNifti+'nii.gz')



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




        if applyWavelet:
              for decompositionImage, decompositionName, inputKwargs in imageoperations.getWaveletImage(image, mask):
                  k = 2
                  w=w+1
                  waveletShapeFeaturs = shape.RadiomicsShape(decompositionImage, mask, **inputKwargs)
                  waveletShapeFeaturs.enableAllFeatures()
                  results = waveletShapeFeaturs.execute()
                  print('Calculated ShapeFeaturs features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                      print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      if w == 1:
                          sheet2 = workbook.add_sheet('Wavelet based image-'+Name[INIT])
                          sheet2.write(0, 0, 'Patient_ID')
                          sheet2.write(0, 1, 'Decomposition Name')

                      sheet2.write(w, 0, str(testcase))
                      sheet2.write(w, 1,str(decompositionName))
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      if w == 1:
                          sheet2_2 = workbook.add_sheet('Wavelet based image-'+Name[INIT+1])
                          sheet2_2.write(0, 0, 'Patient_ID')
                          sheet2_2.write(0, 1, 'Decomposition Name')

                      sheet2_2.write(w, 0, str(testcase))
                      sheet2_2.write(w, 1, str(decompositionName))
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1



                  waveletFirstOrderFeaturs = firstorder.RadiomicsFirstOrder(decompositionImage, mask, **inputKwargs)
                  waveletFirstOrderFeaturs.enableAllFeatures()
                  results = waveletFirstOrderFeaturs.execute()
                  print('Calculated firstorder features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                    waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                    print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1



                  waveletglcmFeatures = glcm.RadiomicsGLCM(decompositionImage, mask, **inputKwargs)
                  waveletglcmFeatures.enableAllFeatures()
                  results = waveletglcmFeatures.execute()
                  print('Calculated glcmFeatures features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                      print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1



                  waveletgldmFeatures = gldm.RadiomicsGLDM(decompositionImage, mask, **inputKwargs)
                  waveletgldmFeatures.enableAllFeatures()
                  results = waveletgldmFeatures.execute()
                  print('Calculated gldmFeatures features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                      print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1


                  waveletglrlmFeatures = glrlm.RadiomicsGLRLM(decompositionImage, mask, **inputKwargs)
                  waveletglrlmFeatures.enableAllFeatures()
                  results = waveletglrlmFeatures.execute()
                  print('Calculated glrlmFeatures features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                      print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1



                  waveletglszmFeatures = glszm.RadiomicsGLSZM(decompositionImage, mask, **inputKwargs)
                  waveletglszmFeatures.enableAllFeatures()
                  results = waveletglszmFeatures.execute()
                  print('Calculated glszmFeatures features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                      print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1


                  waveletngtdmFeatures = ngtdm.RadiomicsNGTDM(decompositionImage, mask, **inputKwargs)
                  waveletngtdmFeatures.enableAllFeatures()
                  results = waveletngtdmFeatures.execute()
                  print('Calculated ngtdmFeatures features with wavelet ', decompositionName)
                  for (key, val) in six.iteritems(results):
                      waveletFeatureName = '%s_%s' % (str(decompositionName), key)
                      print('  ', waveletFeatureName, ':', val)

                  if m==1:
                      for key, val in results.items():
                          if w == 1:
                              sheet2.write(0, k, str(key))
                          sheet2.write(w, k, str(val))
                          k = k + 1
                  if m==2:
                      for key, val in results.items():
                          if w == 1:
                              sheet2_2.write(0, k, str(key))
                          sheet2_2.write(w, k, str(val))
                          k = k + 1



print('Finish')
def SaveAs():
    pathname1 = filedialog.askdirectory(parent=root, initialdir=currdir, title='SaveAs: Select a directory')
    if os.path.isdir(pathname1) != True:
        SaveAs()
    return pathname1

pathname1=SaveAs()
workbook.save(pathname1+'/Results.csv')



