
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtWidgets, uic, Qt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from vtk.util.numpy_support import numpy_to_vtk
from PyQt5.QtWidgets import QApplication, QMainWindow
import matplotlib.pyplot as plt



from renderers.Surface import SurfaceRenderer #render file

global_i = 0


# This function gets the name of the file we are interested and returns the numpy array
def load_data_as_numpy(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    image = reader.Execute();

    array = sitk.GetArrayFromImage(image)

    return array
    
    
#####Multi window
class MainWindow1(QMainWindow):

    def __init__(self, render_list, shape, frame):
        # check if render_list and mayout match
        assert len(render_list) == shape[0] * shape[1]
        
        QMainWindow.__init__(self)

        self.vl = Qt.QVBoxLayout()
        self.horizontalGroupBox = Qt.QGroupBox("Surface rendering")
        self.vl.addWidget(self.horizontalGroupBox)
        self.layout = Qt.QGridLayout()
        self.frame = frame
        self.render = render_list
        self.vtk_widgets = [ren_.interactor for ren_ in self.render]
        self.layout_grid = idx_from_shape(shape)
        
        self.rens = list()
        self.irens = list()
        print("loading...")

        for idx, (wid_, grd_, ren_) in enumerate(zip(self.vtk_widgets,
            self.layout_grid, self.render)):
            self.layout.addWidget(wid_, grd_[0], grd_[1])

            self.rens.append(vtk.vtkRenderer()) 
            wid_.GetRenderWindow().AddRenderer(ren_.renderer)
            self.irens.append(wid_.GetRenderWindow().GetInteractor())

        self.horizontalGroupBox.setLayout(self.layout)
        self.frame.setLayout(self.vl)

        self.setCentralWidget(self.horizontalGroupBox)


        for iren_ in self.irens:
            iren_.Initialize()
            iren_.Start()
            
        


def idx_from_shape(shape_):
        assert len(shape_) == 2
        retlist = list()
        for i_ in range(shape_[0]):
            for j_ in range(shape_[1]):
                retlist.append((i_, j_))

        return retlist


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('LI_main_GUI.ui', self) # 10)
        global global_i
        self.nr_files_main.setText(str(global_i))
        
        self.show()
        #push buttons on main GUI
        self.volume_push.clicked.connect(self.volume_pushed)
        self.surface_push.clicked.connect(self.surface_pushed)
        self.segment_push.clicked.connect(self.file_pushed)
        self.multi_push.clicked.connect(self.start)
        self.fused_push.clicked.connect(self.fused_ui)
        self.Exit_main.clicked.connect(lambda:self.close())
        
        
    def fused_ui(self):
        uic.loadUi('fused.ui', self)
        self.show
        self.render_fused.clicked.connect(self.fused_start)
        self.fuse_return.clicked.connect(lambda:self.close())
        self.fuse_return.clicked.connect(self.__init__)

        
    def fused_start(self):
        # 1 get data path from the first argument given

        filename_lungs = "segmentedlungs.nii"
        filename_torso = "segmentedtorso.nii"
        filename_image = "IMG_0031.nii.gz"




        #Read the lungs
        reader_lungs = vtk.vtkNIFTIImageReader()
        reader_lungs.SetFileName(filename_lungs) # replace filename

        #Read the torso
        reader_torso = vtk.vtkNIFTIImageReader()
        reader_torso.SetFileName(filename_torso)


        volmpt = vtk.vtkGPUVolumeRayCastMapper()
#volmp.SetInputConnection(reader_src.GetOutputPort())
        volmpt.SetInputConnection(reader_torso.GetOutputPort())
        volmpl = vtk.vtkGPUVolumeRayCastMapper()
#volmp.SetInputConnection(reader_src.GetOutputPort())
        volmpl.SetInputConnection(reader_lungs.GetOutputPort())



# 4 transfer functions for color and opacity

#LUNGS


        funAlphal = vtk.vtkPiecewiseFunction() # opacity
        
        r1 = 1
        b1 = 0
        g1 = 0
        r2 = 0
        b2 = 1
        g2 = 0
        op_int = 1
        op_ple = 0.001
        low = 50
        high = 200      
        
        
        funAlphal.AddPoint(0,0)
        funAlphal.AddPoint(1,op_ple)
        funAlphal.AddPoint(-10,op_ple) #pleura divide by 100
        funAlphal.AddPoint(10,op_ple)


        funAlphal.AddPoint(low,op_int) #internals divide by 10
        funAlphal.AddPoint(high,op_int) 

        funAlphal.AddPoint(high + 1 ,0)
        funAlphal.AddPoint(1000,0)


#funAlpha.ClampingOff
        
        funColorl = vtk.vtkColorTransferFunction()
        funColorl.AddRGBPoint(low, r1,g1,b1)
        funColorl.AddRGBPoint(high, r1,g1,b1)

        funColorl.AddRGBPoint(-10, r2,g2,b2)
        funColorl.AddRGBPoint(10, r2,g2,b2)

        funColorl.AddRGBPoint(high + 1 , 0,0.5,0)
        funColorl.AddRGBPoint(1000, 0.5,0,0)




        funColorl.ClampingOff

#TORSO
        funAlphat = vtk.vtkPiecewiseFunction()

        minimum = 2
        maximum = 10

        funAlphat.AddPoint(0,0)
        funAlphat.AddPoint(minimum,0.01)
        funAlphat.AddPoint(maximum,0.01)
        funAlphat.AddPoint(11,0.01)
        funAlphat.AddPoint(50,0.01)
        funAlphat.AddPoint(51,0.01)
        funAlphat.AddPoint(500,0.1)

        funAlphat.ClampingOff()




        funColort = vtk.vtkColorTransferFunction()
        funColort.AddRGBPoint(minimum,0,0,0)
        funColort.AddRGBPoint(maximum,1,1,1)
        funColort.AddRGBPoint(11,0,1,0)
        funColort.AddRGBPoint(50,0,0,1)
        funColort.AddRGBPoint(51,0,0,0)
        funColort.AddRGBPoint(500,0,0,0)

        funColort.ClampingOff()




# 6 set up the volume properties with linear interpolation

        volumePropertyl = vtk.vtkVolumeProperty()

        volumePropertyl.SetColor(0,funColorl)
        volumePropertyl.SetScalarOpacity(0,funAlphal)

        volumePropertyt = vtk.vtkVolumeProperty()

        volumePropertyt.SetColor(0,funColort)
        volumePropertyt.SetScalarOpacity(0,funAlphat)



        volumePropertyl.SetInterpolationTypeToLinear()
        volumePropertyt.SetInterpolationTypeToLinear()


# 7 set up the actor and connect it to the mapper

        volActl = vtk.vtkVolume()
        volActl.SetMapper(volmpl)
        volActl.SetProperty(volumePropertyl)

        volActt = vtk.vtkVolume()
        volActt.SetMapper(volmpt)
        volActt.SetProperty(volumePropertyt)


# 8 set up the camera and the renderer


        renderer = vtk.vtkRenderer()
        camera = vtk.vtkCamera() 

        camera.SetViewUp(0,0,5)
        camera.SetFocalPoint(250,250,250)
        camera.SetPosition(-800,150,100)



# 9 set the color of the renderers background 
        renderer.SetBackground(1.,1.,1.)


# 10 set the renderers camera as active
        renderer.SetActiveCamera(camera)

# 11 add the volume actor to the renderer

#if for actt

        if int(self.lung_slider.value()) == 1:
           renderer.AddActor(volActl)
           
        if int(self.torso_slider.value()) == 1:
           renderer.AddActor(volActt)

# 12 create a render window
        ren_win = vtk.vtkRenderWindow()

# 13 add renderer to the render window
        ren_win.AddRenderer(renderer)

# 14 create an interactor
        iren = vtk.vtkRenderWindowInteractor()

# 15 connect interactor to the render window
        iren.SetRenderWindow(ren_win)
        
# 16 start displaying the render window

        ren_win.Render()

#17 make the window interactive (start the interactor)

        iren.Start()

		
    def start(self):    # 1 read filenames
        
       global global_i
        
       uic.loadUi('Multi_sliders.ui', self)
       self.show
       
       self.global_files_label.setText(str(global_i))
       
       self.files_slider.valueChanged.connect(self.files_slider_changed)  
       self.AmbientSlider.valueChanged.connect(self.amb_s_changed)
       self.SpecSlider.valueChanged.connect(self.spec_s_changed)
       self.DiffSlider.valueChanged.connect(self.diff_s_changed)
       self.OpacSlider.valueChanged.connect(self.opac_s_changed)
       self.RedSlider.valueChanged.connect(self.red_s_changed)
       self.GreenSlider.valueChanged.connect(self.green_s_changed)
       self.BlueSlider.valueChanged.connect(self.blue_s_changed)
       
       
       self.apply_push.clicked.connect(self.apply_changes)
       self.multi_return.clicked.connect(self.apply_changes)
       
       self.multi_return.clicked.connect(lambda:self.close())
       self.multi_return.clicked.connect(self.close_window)
        
        
    def close_window(self):
        
        self.__init__()
        
    def apply_changes(self):
        a =  int(self.OpacSlider.value()) / 100
        b =  int(self.AmbientSlider.value()) / 100
        c =  int(self.DiffSlider.value()) / 100
        d =  int(self.SpecSlider.value()) /100
        e =  int(self.RedSlider.value()) 
        f =  int(self.BlueSlider.value()) - 20
        g =  int(self.GreenSlider.value()) 


        
        
        sliders_input = [a,b,c,d,e,f,g]
        self.apply_done_label.setText("All properties saved, ready to render!")
        self.render_multi_push.clicked.connect(lambda: self.multi_render(sliders_input))
        self.render_multi_push.clicked.connect(self.start)


         

       ###multi render
    def multi_render(self, sliders_input):   
            
            global global_i
            render = list()
            frame = Qt.QFrame()
            
            
            self.apply_done_label_2.setText("Loading...")
            
            if int(self.files_slider.value()) == 1:
                filename0 = 'segmentedlungs.nii.gz'
                #3 define a layout
                layout = (1, 1)
                
                render.append(SurfaceRenderer(filename0, sliders_input, frame=frame))

                
            if int(self.files_slider.value()) == 2:
                filename0 = 'segmentedlungs.nii.gz'
                filename1 = 'segmentedlungs1.nii.gz'
                #3 define a layout
                layout = (1, 2)
                
                render.append(SurfaceRenderer(filename0, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename1, sliders_input, frame=frame))

                
            if int(self.files_slider.value()) == 3:
                filename0 = 'segmentedlungs.nii.gz'
                filename1 = 'segmentedlungs1.nii.gz'
                filename2 = 'segmentedlungs2.nii.gz'
                #3 define a layout
                layout = (1, 3)
                
                render.append(SurfaceRenderer(filename0, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename1, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename2, sliders_input, frame=frame))
                
                
            if int(self.files_slider.value()) == 4:
                filename0 = 'segmentedlungs.nii.gz'
                filename1 = 'segmentedlungs1.nii.gz'
                filename2 = 'segmentedlungs2.nii.gz'
                filename3 = 'segmentedlungs3.nii.gz'
                #3 define a layout
                layout = (1, 4)
                
                render.append(SurfaceRenderer(filename0, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename1, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename2, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename3, sliders_input, frame=frame))

                
            if int(self.files_slider.value()) == 5:
                filename0 = 'segmentedlungs.nii.gz'
                filename1 = 'segmentedlungs1.nii.gz'
                filename2 = 'segmentedlungs2.nii.gz'
                filename3 = 'segmentedlungs3.nii.gz'
                filename4 = 'segmentedlungs4.nii.gz'
                #3 define a layout
                layout = (1, 5)
                
                render.append(SurfaceRenderer(filename0, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename1, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename2, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename3, sliders_input, frame=frame))
                render.append(SurfaceRenderer(filename4, sliders_input, frame=frame))

              
            else:
                
                lambda:self.close()
                self.start()


        #4 render the window
            self.window = MainWindow1(render, layout, frame)
            self.window.show()


        
    def idx_from_shape(shape_):
        assert len(shape_) == 2
        retlist = list()
        for i_ in range(shape_[0]):
            for j_ in range(shape_[1]):
                retlist.append((i_, j_))

        return retlist

        
        
#Get files for segmentation
    def file_pushed(self): 
        uic.loadUi('File_buttons.ui', self)
        
        self.pushButtonDone.clicked.connect(self.pushButtonDone_handler)
        self.nr_slider.valueChanged.connect(self.files_changed)
        self.return_file.clicked.connect(lambda:self.close())
        self.return_file.clicked.connect(self.__init__)

        
    def files_changed(self):
        new_value = str(self.nr_slider.value()) #to show string
        red_int = int(self.nr_slider.value()) #to get int value
        self.files_label.setText(new_value)
        
        
        # A path to the lung image .nii image:
    def pushButtonDone_handler(self):
             self.pushButtonDone.hide()
             self.nr_slider.hide()
             self.label.hide()
             
             global global_i

             file_int = int(self.nr_slider.value()) #to get int value
             i = global_i
             
             for i in range(file_int):
				   
                   print("get image " + str(i+1))
                   filename_image = QFileDialog.getOpenFileName()
                   path_image = filename_image[0]
                   print(path_image)
        
                   print("get mask " + str(i+1))
                   filename = QFileDialog.getOpenFileName()
                   path_mask = filename[0]
                   print(path_mask)
        
                   path_image = filename_image[0]
                   path_mask = filename[0]
		
                   image = path_image
                   mask = path_mask
		
                   #convert image and mask to numpy array
                   image_arr = load_data_as_numpy(image)
                   mask_arr = load_data_as_numpy(mask)
                   #perform the segmentation with vectorization method
                   segmented_arr = np.multiply(mask_arr/255,image_arr)
                   
                   segmented_torso = np.subtract(image_arr, segmented_arr)

	               #Try create niftii file
                   print("loading...")
                   if i == 0:
                        newimage = nib.Nifti1Image(segmented_arr, affine=np.eye(4))
                        nib.loadsave.save(newimage,'segmentedlungs.nii.gz')
                        global_i = i+1
                        
                        newimagetorso = nib.Nifti1Image(segmented_torso, affine=np.eye(4))
                        nib.loadsave.save(newimagetorso, 'segmentedtorso.nii.gz')
	               
                   if i == 1:
                        newimage = nib.Nifti1Image(segmented_arr, affine=np.eye(4))
                        nib.loadsave.save(newimage,'segmentedlungs1.nii.gz')
                        global_i = i+1
	               
                   if i == 2:
                        newimage = nib.Nifti1Image(segmented_arr, affine=np.eye(4))
                        nib.loadsave.save(newimage,'segmentedlungs2.nii.gz')
                        global_i = i+1
	               
                   if i == 3:
                        newimage = nib.Nifti1Image(segmented_arr, affine=np.eye(4))
                        nib.loadsave.save(newimage,'segmentedlungs3.nii.gz')
                        global_i = i+1
	               
                   if i == 4:
                        newimage = nib.Nifti1Image(segmented_arr, affine=np.eye(4))
                        nib.loadsave.save(newimage,'segmentedlungs4.nii.gz')
                        global_i = i+1
                        
                        
                   print ("Number of files: " +str(i+1))
                   
                   print("Return to Main window")
                   
                   self.files_label.setText("Segmentation " +str(i+1)+ " out of " +str(file_int) +" is done") 

             self.files_label.setText("Segmentation Has Finished!")
             self.return_file.clicked.connect(lambda:self.close())
             self.return_file.clicked.connect(self.__init__)
        
        
        # SURFACE RENDERING
    def surface_pushed(self):
        uic.loadUi('Surface_sliders.ui', self)
        self.show()
        
        #check sliders
        self.AmbientSlider.valueChanged.connect(self.amb_s_changed)
        self.SpecSlider.valueChanged.connect(self.spec_s_changed)
        self.DiffSlider.valueChanged.connect(self.diff_s_changed)
        self.OpacSlider.valueChanged.connect(self.opac_s_changed)
        self.RedSlider.valueChanged.connect(self.red_s_changed)
        self.GreenSlider.valueChanged.connect(self.green_s_changed)
        self.BlueSlider.valueChanged.connect(self.blue_s_changed)
        
        #check buttons
        self.surface_push.clicked.connect(self.surface_render_handler) 


        self.surface_return.clicked.connect(lambda:self.close())
        self.surface_return.clicked.connect(self.__init__)
        

        
        
    def files_slider_changed(self):
         new_value = str(self.files_slider.value()) #to show string
         amb_int = int(self.AmbientSlider.value()) #to get int value
         self.files_label.setText(new_value)            
        
        
    def amb_s_changed(self):
         new_value = str(self.AmbientSlider.value()) #to show string
         amb_int = int(self.AmbientSlider.value()) #to get int value
         self.ambience_label.setText(new_value)
            
         
    def spec_s_changed(self):
         new_value = str(self.SpecSlider.value()) #to show string
         spec_int = int(self.SpecSlider.value()) #to get int value
         self.spec_label.setText(new_value) 
  
    def diff_s_changed(self):
         new_value = str(self.DiffSlider.value()) #to show string
         diff_int = int(self.DiffSlider.value()) #to get int value
         self.diff_label.setText(new_value)
         
    def opac_s_changed(self):
         new_value = str(self.OpacSlider.value()) #to show string
         opac_int = int(self.OpacSlider.value()) #to get int value
         self.opac_label.setText(new_value)
         
    def red_s_changed(self):
         new_value = str(self.RedSlider.value()) #to show string
         red_int = int(self.RedSlider.value()) #to get int value
         self.red_label.setText(new_value)
         
         
    def blue_s_changed(self):
         new_value = str(self.BlueSlider.value()) #to show string
         blue_int = int(self.BlueSlider.value()) #to get int value
         self.blue_label.setText(new_value)
         
    def green_s_changed(self):
         new_value = str(self.GreenSlider.value()) #to show string
         green_int = int(self.GreenSlider.value()) #to get int value
         self.green_label.setText(new_value)


              ####animation
    def ani_start(self):
        filename = "segmentedlungs.nii.gz"
        
        
        
        # 2 set up the source
        reader_src = vtk.vtkNIFTIImageReader() #this class reads Nifti files (medical images)
        reader_src.SetFileName(filename) #here we should read the name of our medical image file in the class and put in the class

        # set the origin of the data to its center 
        reader_src.Update()
        data = reader_src.GetOutput() # Get the output data object for a port on this algorithm.
        center = data.GetCenter() #Get the center of the bounding box of the dataset.
        data.SetOrigin(- center[0], - center[0], - center[0]) #Set the origin of the image dataset.

        # 3 (filter) 
        cast_filter = vtk.vtkImageCast() #Image data type casting filter: casts the input type to match the output type in the image processing pipeline

        cast_filter.SetInputConnection(reader_src.GetOutputPort()) #SetInputData(..) assign a data object as input. Note that this method does not establish a pipeline connection
													
															
        cast_filter.SetOutputScalarTypeToUnsignedShort() #set the desired output scalar type to cast to

        # 4 marching cubes (mapper)
        contour = vtk.vtkMarchingCubes() #generate isosurface(s) from volume. It is a filter that takes as input a volume (e.g., 3D structured point set) and generates on
								 #output one or more isosurfaces. One or more contour values must be specified to generate the isosurfaces. Alternatively, you can
								 #specify amin/max scalar range and the number of contours to generate a series of evenlu spaced contour values.


        contour.SetInputConnection(cast_filter.GetOutputPort()) #Here we connect the input to the output?
        contour.ComputeNormalsOff() #set/get the computation of normals. Normal computation is fairly expensice in both time and storage. If the output data will be processed
						   #filters that modify topology or geometry, it may be wise to turn Normals and Gradients off.
        contour.ComputeGradientsOff() #set/get the computation of graients. Gradient computation is fairly expensive in both time and storage. Note that id ComputeNormals
							 #is on, gradients will have to be calculated, but will nmot be stored in the output dataset. If the output data will be processed by
							 #the filters that modify topology or geometry, it may be wise to turn Normals and Gradients off.
        contour.SetNumberOfContours(2)
        contour.SetValue(0, 100) 
        contour.SetValue(1, 100) 


        # 4.5 Decimation

       
        deci = vtk.vtkDecimatePro()

     
        deci.SetInputConnection(contour.GetOutputPort())
        deci.SetTargetReduction(0.9)
        deci.PreserveTopologyOff()

        # 4.5 Smoothing

    
        smoother = vtk.vtkWindowedSincPolyDataFilter()
     
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(deci.GetOutputPort())
        smoother.SetNumberOfIterations(500)

        #Normals if Off in Marching cubes
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.FlipNormalsOn()



        con_mapper = vtk.vtkPolyDataMapper() #Uses a OpenGL to do the actual rendering


        con_mapper.SetInputConnection(contour.GetOutputPort()) #Get the output port of the contour contour and connects it with input
        con_mapper.ScalarVisibilityOff()


        # 5 set up the actor

        #set properties for our actor
       
        prop = vtk.vtkProperty()


        opac = 0.2
        amb = 0.2
        diff = 0.2
        spec = 0.2
        specpwr = 1

        red = 0.6
        green = 0.2
        blue = 0.2

        prop.SetOpacity (opac)
        prop.SetAmbient (amb)
        prop.SetDiffuse (diff)
        prop.SetSpecular (spec)
        prop.SetSpecularPower (specpwr)

        prop.SetColor (red,blue,green)

        

        #con_mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor() # a concrete implementattion of the abstract class vtkActor is different. 
        actor.SetMapper(con_mapper) # this is the method that is used to connect an actor to the end of a visualization pipeline, i.e. the mapper. 
        actor.SetProperty(prop) #assign properties to our actor


        # 6 set up the camera and the renderer
        renderer = vtk.vtkRenderer() 

        camera = vtk.vtkCamera() #this I suppose is the camera
        camera.SetViewUp(0., 1., 0.)
        camera.SetPosition(-500, 100, 100)
        camera.SetFocalPoint(0, 0, 0)

        # 7 set the color of the renderers background to black (0., 0., 0.)

       
        renderer.SetBackground(1, 1, 1)

        # 8 set the renderers canera as active
        renderer.SetActiveCamera(camera)

        # 9 add the volume actor to the renderer
        renderer.AddActor(actor)
        
        


        
        # 10 create a render window
        ren_win = vtk.vtkRenderWindow()

        # 11 add renderer to the render window
        ren_win.AddRenderer(renderer)

        # 12 create an interactor
        iren = vtk.vtkRenderWindowInteractor()

        # 13 connect interactor to the render window
        iren.SetRenderWindow(ren_win)

#  start displaying the render window
        ren_win.Render()

# animation 

        iren.Initialize() # need to initialize our interactor before we can add things. Initializes the event handlers without an XtAppContext.
				  #This is good for when you don't have a user interface, but you still want to have mouse interaction.

        timer_call = TimerCallback(actor)
        iren.AddObserver('TimerEvent', timer_call.execute) #Add an event callback function(vtkObject, int) for an event type. Returns a handle that can be used with RemoveEvent(int).

        iren.CreateRepeatingTimer(10) #Create a repeating timer, with the specified duration (in milliseconds). Return the timer id.



        #make the window interactive
        iren.Start()

         ##surface rendering
    def surface_render_handler(self):

        filename = "segmentedlungs.nii.gz"
        
        
        
        # 2 set up the source
        reader_src = vtk.vtkNIFTIImageReader() #this class reads Nifti files (medical images)
        reader_src.SetFileName(filename) #here we should read the name of our medical image file in the class and put in the class

        # 3 (filter) 
        cast_filter = vtk.vtkImageCast() #Image data type casting filter: casts the input type to match the output type in the image processing pipeline
								
								
        cast_filter.SetInputConnection(reader_src.GetOutputPort()) #SetInputData(..) assign a data object as input. Note that this method does not establish a pipeline connection
														
															
        cast_filter.SetOutputScalarTypeToUnsignedShort() #set the desired output scalar type to cast to

        # 4 marching cubes (mapper)
        contour = vtk.vtkMarchingCubes() #generate isosurface(s) from volume. It is a filter that takes as input a volume (e.g., 3D structured point set) and generates on
								

        contour.SetInputConnection(cast_filter.GetOutputPort()) #Here we connect the input to the output?
        contour.ComputeNormalsOff() #set/get the computation of normals. Normal computation is fairly expensice in both time and storage. If the output data will be processed
        contour.ComputeGradientsOff() #set/get the computation of graients. Gradient computation is fairly expensive in both time and storage. Note that id ComputeNormals

        contour.SetValue(0, 100) 


        # 4.5 Decimation

 
        deci = vtk.vtkDecimatePro()

        deci.SetInputConnection(contour.GetOutputPort())
        deci.SetTargetReduction(0.9)
        deci.PreserveTopologyOff()

        # 4.5 Smoothing


        smoother = vtk.vtkWindowedSincPolyDataFilter()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(deci.GetOutputPort())
        smoother.SetNumberOfIterations(500)

        #Normals if Off in Marching cubes
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.FlipNormalsOn()



        con_mapper = vtk.vtkPolyDataMapper() #Uses a OpenGL to do the actual rendering


        con_mapper.SetInputConnection(contour.GetOutputPort()) #Get the output port of the contour contour and connects it with input
        con_mapper.ScalarVisibilityOff()


        # 5 set up the actor

        #set properties for our actor

        prop = vtk.vtkProperty()


        opac = int(self.OpacSlider.value()) / 100
        amb = int(self.AmbientSlider.value()) / 100
        diff = int(self.DiffSlider.value()) / 100
        spec = int(self.SpecSlider.value()) / 100
        specpwr = 1

        red = int(self.RedSlider.value()) / 100
        green = int(self.GreenSlider.value()) / 100
        blue = int(self.BlueSlider.value()) / 100

        prop.SetOpacity (opac)
        prop.SetAmbient (amb)
        prop.SetDiffuse (diff)
        prop.SetSpecular (spec)
        prop.SetSpecularPower (specpwr)

        prop.SetColor (red,green,blue)

        

        actor = vtk.vtkActor() # a concrete implementattion of the abstract class vtkActor is different. 
        actor.SetMapper(con_mapper) # this is the method that is used to connect an actor to the end of a visualization pipeline, i.e. the mapper. 
        actor.SetProperty(prop) #assign properties to our actor


        # 6 set up the camera and the renderer
        renderer = vtk.vtkRenderer() 

        camera = vtk.vtkCamera() 
        camera.SetViewUp(0., 1., 0.)
        camera.SetPosition(-500, 100, 100)
        camera.SetFocalPoint(200, 200, 200)

        # 7 set the color of the renderers background to black (0., 0., 0.)


        renderer.SetBackground(1, 1, 1)

        # 8 set the renderers canera as active
        renderer.SetActiveCamera(camera)

        # 9 add the volume actor to the renderer
        renderer.AddActor(actor)
        
        


        
        # 10 create a render window
        ren_win = vtk.vtkRenderWindow()

        # 11 add renderer to the render window
        ren_win.AddRenderer(renderer)

        # 12 create an interactor
        iren = vtk.vtkRenderWindowInteractor()

        # 13 connect interactor to the render window
        iren.SetRenderWindow(ren_win)
        
        if int(self.ani_slider.value()) == 1:
           
            self.ani_start()
            
        if int(self.stereo_slider.value()) == 1:
            print("stereo enbabled")
            ren_win.GetStereoCapableWindow()
            ren_win.StereoCapableWindowOn()
            ren_win.SetStereoRender(1)
            ren_win.SetStereoTypeToCrystalEyes()
            ren_win.Render()
            iren.Start()

        else:
			 
         # 14 start displaying the render window
          
          ren_win.Render()
        

         # 15 make the window interactive
          iren.Start()
        
        
#VOLUME RENDERING


    def volume_pushed(self):
        uic.loadUi('volume_rendering.ui', self)
		
        self.show()
        self.return_push.clicked.connect(lambda:self.close())
        self.return_push.clicked.connect(self.__init__)

        #check if slider changes
        self.r_slider.valueChanged.connect(self.red_changed)
        self.b_slider.valueChanged.connect(self.blue_changed)
        self.g_slider.valueChanged.connect(self.green_changed)
        self.r_slider_2.valueChanged.connect(self.red2_changed)
        self.b_slider_2.valueChanged.connect(self.blue2_changed)
        self.g_slider_2.valueChanged.connect(self.green2_changed)
        self.opacity_int.valueChanged.connect(self.opacity_int_changed)
        self.opacity_ple.valueChanged.connect(self.opacity_ple_changed)
        self.lower.valueChanged.connect(self.lower_changed)
        self.higher.valueChanged.connect(self.higher_changed)
        
        self.render_done.clicked.connect(self.render_handler)
        
        
    def red_changed(self):
            new_value = str(self.r_slider.value()) #to show string
            red_int = int(self.r_slider.value()) #to get int value
            self.r_label.setText(new_value)
            
    def blue_changed(self):
            new_value = str(self.b_slider.value()) #to show string
            blue_int = int(self.b_slider.value()) #to get int value
            self.b_label.setText(new_value)
            
    def green_changed(self):
            new_value = str(self.g_slider.value()) #to show string
            g_int = int(self.g_slider.value()) #to get int value
            self.g_label.setText(new_value)
            
    def red2_changed(self):
            new_value = str(self.r_slider_2.value()) #to show string
            red2_int = int(self.r_slider_2.value()) #to get int value
            self.r_label_2.setText(new_value)
            
    def blue2_changed(self):
            new_value = str(self.b_slider_2.value()) #to show string
            blue2_int = int(self.b_slider_2.value()) #to get int value
            self.b_label_2.setText(new_value)
            
    def green2_changed(self):
            new_value = str(self.g_slider_2.value()) #to show string
            green2_int = int(self.g_slider_2.value()) #to get int value
            self.g_label_2.setText(new_value)
            
    def opacity_int_changed(self):
            new_value = str(self.opacity_int.value()) #to show string
            opacity_int_int = int(self.opacity_int.value()) #to get int value
            self.opacity_int_label.setText(new_value)
            
    def opacity_ple_changed(self):
            new_value = str(self.opacity_ple.value()) #to show string
            opacity_ple_int = int(self.opacity_ple.value()) #to get int value
            self.opacity_ple_label.setText(new_value)
            
    def lower_changed(self):
            new_value = str(self.lower.value()) #to show string
            lower_int = int(self.lower.value()) #to get int value
            self.lower_2.setText(new_value)
            
    def higher_changed(self):
            new_value = str(self.higher.value()) #to show string
            higher_int = int(self.higher.value()) #to get int value
            self.lower_3.setText(new_value)
            
  
        ###volume render
    def render_handler(self):
        import vtk
        import sys



        filename = "segmentedlungs.nii"


        reader_src = vtk.vtkNIFTIImageReader()
        reader_src.SetFileName(filename) # replace filename




        # 3 set up the volume mapper
        volmp = vtk.vtkGPUVolumeRayCastMapper()
        volmp.SetInputConnection(reader_src.GetOutputPort())



        # 4 transfer functions for color and opacity




        funAlpha = vtk.vtkPiecewiseFunction() # opacity
        
        
        r1 = int(self.r_slider.value()) / 100
        b1 = int(self.b_slider.value()) / 100
        g1 = int(self.g_slider.value()) / 100
        r2 = int(self.r_slider_2.value()) / 100
        b2 = int(self.b_slider_2.value()) / 100
        g2 = int(self.g_slider_2.value()) / 100
        op_int = int(self.opacity_int.value()) /100
        op_ple = int(self.opacity_ple.value()) / 1000
        low = int(self.lower.value()) 
        high = int(self.higher.value()) 
        
        
        
        funAlpha.AddPoint(0,0)

        funAlpha.AddPoint(-10,op_ple) #pleura divide by 100
        funAlpha.AddPoint(10,op_ple)


        funAlpha.AddPoint(low,op_int) #internals divide by 10
        funAlpha.AddPoint(high,op_int)

        funAlpha.AddPoint(high + 1 ,0)
        funAlpha.AddPoint(1000,0)


      
        funColor = vtk.vtkColorTransferFunction()
        funColor.AddRGBPoint(low, r1,g1,b1)
        funColor.AddRGBPoint(high, r1,g1,b1)

        funColor.AddRGBPoint(-10, r2,g2,b2)
        funColor.AddRGBPoint(10, r2,g2,b2)

        funColor.AddRGBPoint(high + 1 , 0,0.5,0)
        funColor.AddRGBPoint(1000, 0.5,0,0)




        funColor.ClampingOff


        # 6 set up the volume properties with linear interpolation

        volumeProperty = vtk.vtkVolumeProperty()

        volumeProperty.SetColor(0,funColor)
        volumeProperty.SetScalarOpacity(0,funAlpha)


        #volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationTypeToLinear()



        # 7 set up the actor and connect it to the mapper

        volAct = vtk.vtkVolume()
        volAct.SetMapper(volmp)
        volAct.SetProperty(volumeProperty)



        renderer = vtk.vtkRenderer()
        camera = vtk.vtkCamera() 

        camera.SetViewUp(0.,1.,0.)
        camera.SetFocalPoint(250,250,250)
        camera.SetPosition(-600,150,100)



        # 9 set the color of the renderers background to black (0., 0., 0.)
        renderer.SetBackground(1.,1.,1.)


        # 10 set the renderers camera as active
        renderer.SetActiveCamera(camera)

        # 11 add the volume actor to the renderer
        renderer.AddActor(volAct)

        # 12 create a render window
        ren_win = vtk.vtkRenderWindow()

        # 13 add renderer to the render window
        ren_win.AddRenderer(renderer)

        # 14 create an interactor
        iren = vtk.vtkRenderWindowInteractor()

        # 15 connect interactor to the render window
        iren.SetRenderWindow(ren_win)
        
        # 16 start displaying the render window

        if int(self.stereo_volume.value()) == 1:
 
            ren_win.GetStereoCapableWindow()
            ren_win.StereoCapableWindowOn()
            ren_win.SetStereoRender(1)
            ren_win.SetStereoTypeToAnaglyph()
        
        ren_win.Render()

        # 17 make the window interactive (start the interactor)
        iren.Start()
        
    
class TimerCallback():
        
        
	
        def __init__(self, actor):
            self.actor = actor
            self.timer_count = 0
		
        def get_angle(self):
			#angle = timer_count % 360
            self.angle = self.timer_count % 360
            return self.angle
		
        def execute(self,obj,event):
			
            
            self.n_angle = self.get_angle()
            
            self.actor.SetOrientation(0,0,self.n_angle) #rotate around z axes
			#self.actor.SetOrientation(0,self.n_angle,0) #rotate around y axes
			#self.actor.SetOrientation(self.n_angle,0,0) #rotate around x axes
			
            obj.GetRenderWindow().Render()
            self.timer_count += 1
            #by changing this we are changing the velocity of our rotation
		    
def load_data_as_numpy(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    image = reader.Execute();

    array = sitk.GetArrayFromImage(image)

    return array

        
        
class start_GUI():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()
    
start_GUI()
           
        

