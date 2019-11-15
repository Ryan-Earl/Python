import arcpy

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [Unique_Value]


class Unique_Value(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Tool"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        param0 = arcpy.Parameter(
            displayName = 'Pro Project Path',
            name = 'projpath',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        param1 = arcpy.Parameter(
            displayName = 'Layer Name',
            name = 'lyrname',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        param2 = arcpy.Parameter(
            displayName = 'Output project name and path',
            name = 'output name',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        #Define progressor
        readTime = 2.5
        start = 0
        maximum = 100
        step = 50

        #setup progressor
        arcpy.SetProgressor("step", "applying symbology...", start, maximum, step)
        time.sleep(readTime)
        arcpy.AddMessage("applying symbology...")

        pro_path = parameters[0].valueAsText
        project = arcpy.mp.ArcGISProject(pro_path)
        campus = project.listMaps('Map')[0]

        for lyr in campus.listLayers():
            if lyr.isFeatureLayer:
                symbology = lyr.symbology
                if hasattr(symbology, 'renderer'):
                    if lyr.name == parameters[1].valueasText:
                        symbology.updateRenderer("UniqueValueRenderer")
                        symbology.renderer.fields = ["LotType"]
                        lyr.symbology = symbology
                    else:
                        print("NOT GarageParking")

        arcpy.SetProgressorPosition(start + step)
        arcpy.SetProgressorLabel("saving new project...")
        time.sleep(readTime)
        arcpy.AddMessage("saving new project...")

        project.saveACopy(parameters[2].valueAsText)
        return