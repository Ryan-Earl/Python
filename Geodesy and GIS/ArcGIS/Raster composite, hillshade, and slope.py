import arcpy

# RS Composite
path = r"E:\\Random GIS\\Raster Code"
band1 = arcpy.sa.Raster(path + "\\LT05_L1TP_026039_20110819_20160831_01_T1_B1.TIF") # blue
band2 = arcpy.sa.Raster(path + "\\LT05_L1TP_026039_20110819_20160831_01_T1_B2.TIF") # green
band3 = arcpy.sa.Raster(path + "\\LT05_L1TP_026039_20110819_20160831_01_T1_B3.TIF") # red
band4 = arcpy.sa.Raster(path + "\\LT05_L1TP_026039_20110819_20160831_01_T1_B4.TIF") # NIR
composite = arcpy.CompositeBands_management([band1, band2, band3, band4], path + "\\composite.TIF")

# Hillshade
path = r"E:\\Random GIS\\Raster Code"
azimuth = 315
altitude = 45
shadows = "NO_SHADOWS"
z_factor = 1
arcpy.ddd.HillShade(path + r"\\n30_w097_1arc_v3.tif", path + r"\\hillshade.tif", azimuth, altitude, shadows, z_factor)

# Slope
output_measurement = "DEGREE"
z_factor = 1
# method = "PLANAR"
# z_unit = "METER"
arcpy.ddd.Slope(path + r"\\n30_w097_1arc_v3.tif", path + r"\\slope.tif", output_measurement, z_factor)