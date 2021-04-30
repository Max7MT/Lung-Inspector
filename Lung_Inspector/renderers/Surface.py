

import os
import sys

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class SurfaceRenderer:
    def __init__(self, filename, properties, frame):
        # load the data (source)
        reader_src = vtk.vtkNIFTIImageReader()
        reader_src.SetFileName(filename)

        # filter 
        cast_filter = vtk.vtkImageCast()
        cast_filter.SetInputConnection(reader_src.GetOutputPort())
        cast_filter.SetOutputScalarTypeToUnsignedShort()

        # marching cubes (mapper)
        contour = vtk.vtkMarchingCubes()
        contour.SetInputConnection(cast_filter.GetOutputPort())
        contour.ComputeNormalsOn()
        contour.ComputeGradientsOn()
        contour.SetValue(0, 100)
        
        con_mapper =vtk.vtkPolyDataMapper()
        con_mapper.SetInputConnection(contour.GetOutputPort())
        
        
        prop = vtk.vtkProperty()


        
        opac = properties[0]
        amb = properties[1]
        diff = properties[2]
        spec = properties[3]
        specpwr = 1

        red = properties[4]
        blue = properties[5]
        green = properties[6]
        

        prop.SetOpacity (opac)
        prop.SetAmbient (amb)
        prop.SetDiffuse (diff)
        prop.SetSpecular (spec)
        prop.SetSpecularPower (specpwr)

        prop.SetColor (red,green,blue)
        
       
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(con_mapper)
        
        actor.SetProperty(prop) 

        # setup the camera and the renderer
        self.renderer = vtk.vtkRenderer()

        camera = self.renderer.MakeCamera()
        camera.SetViewUp(0., 0., -.1)
        camera.SetPosition(-400, 100, 100)

        self.renderer.SetBackground(1., 1., 1.) # so to white
        self.renderer.SetActiveCamera(camera)
        self.renderer.AddActor(actor) 

        # window interaction (camera movement etc)
        self.interactor = QVTKRenderWindowInteractor(frame)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        self.interactor.Initialize()


