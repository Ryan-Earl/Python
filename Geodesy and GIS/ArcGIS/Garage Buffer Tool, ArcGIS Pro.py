import arcpy

class Toolbox(object):
    def __init__(self):
        self.label = "Toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [GarageProximity]


class GarageProximity(object):
    def __init__(self):
        self.label = "Garage Proximity"
        self.description = "Finds the intersection of the building closest to a given garage"
        self.canRunInBackground = False

    def getParameterInfo(self):
        param0 = arcpy.Parameter(
            displayName = "GDB Folder Path",
            name = "GDBFolder",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Input"
        )
        param1 = arcpy.Parameter(
            displayName = 'GDB Name',
            name = 'GDBName',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        param2 = arcpy.Parameter(
            displayName = 'Campus GDB Path',
            name = 'CampusGDB',
            datatype = 'DEType',
            parameterType = 'Required',
            direction = 'Input' 
        )
        param3 = arcpy.Parameter(
            displayName = 'Garage CSV File',
            name = 'CSVFile',
            datatype = 'DEFile',
            parameterType = 'Required',
            direction  = 'Input'
        )
        param4 = arcpy.Parameter(
            displayName = 'Name of Created Garage Layer',
            name = 'GarageLayerName',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        param5 = arcpy.Parameter(
            displayName = 'Name of Garage you wish to buffer',
            name = 'SelectedGarageName',
            datatype = 'GPString',
            parameterType = 'Required',
            direction = 'Input'
        )
        param6 = arcpy.Parameter(
            displayName = 'Buffer Distance (meters)',
            name = 'BufferDistance',
            datatype = 'GPDouble',
            parameterType = 'Required',
            direction = 'Input'
        )
        params = [param0, param1, param2, param3, param4, param5, param6]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        # create GDB
        folder = parameters[0].valueAsText
        name = parameters[1].valueAsText
        arcpy.CreateFileGDB_management(folder, name)
        gdb_path = folder + '\\' + name

        # create garages shapefile, add to GDB
        garage_location = parameters[3].valueAsText
        garage_shp_name = parameters[4].valueAsText
        garages = arcpy.MakeXYEventLayer_management(garage_location, 'X', 'Y', garage_shp_name)
        arcpy.FeatureClassToGeodatabase_conversion(garages, gdb_path)
        garage_path = gdb_path + '\\' + garage_shp_name

        # create buildings shapefile given the structures .shp in Campus
        campus_gdb_path = parameters[2].valueAsText
        structures = campus_gdb_path + '\Structures'
        campus_buildings = gdb_path + '\\' + 'campus_building'
        arcpy.Copy_management(structures, campus_buildings)

        # reproject garages to the spatial reference of campus buildings
        projection = arcpy.Describe(campus_buildings).spatialReference
        arcpy.Project_management(garage_path, gdb_path + '\garage_projected', projection)
        garage_projected = gdb_path + '\garage_projected'

        # get building to buffer and buffer distance
        garage_selection = parameters[5].valueAsText
        buffer_distance = float(parameters[6].valueAsText)

        # make sure garage exists
        where = "Name = '%s'" % garage_selection
        cursor = arcpy.SearchCursor(garage_projected, where_clause=where)
        shouldProceed = False
        for row in cursor:
            if row.getValue('Name') == garage_selection:
                shouldProceed = True

        # if should proceed = true
        if shouldProceed:
            # generate the name for buffer layer
            garage_buff = r'\garage_%s_buffed_%s' % (garage_selection, buffer_distance)

            # get reference to building
            garageFeature = arcpy.Select_analysis(garage_projected, gdb_path + r'building_%s' % (garage_selection), where)

            # buffer selected garage
            garage_buffered = arcpy.Buffer_analysis(garageFeature, gdb_path + garage_buff, buffer_distance)

            # intersection of garage buffer and campus buildings
            arcpy.Intersect_analysis([gdb_path + garage_buff, gdb_path + r'\campus_buildings'], gdb_path + '\garage_building_intersection', 'All')

            # convert to csv
            arcpy.TableToTable_conversion(gdb_path + '\garage_building_intersection.dbf', 'C:\\Users\\Eileen\\Documents\\lab 5', 'nearbyBuildings.csv')
        else:
            messages.addErrorMessage('garage not found')
            raise arcpy.ExecuteError
        return
