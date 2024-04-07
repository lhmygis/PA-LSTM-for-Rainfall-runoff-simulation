import os
import pandas as pd
import numpy as np
from datetime import datetime

class DataforIndividual():
    def __init__(self, working_path, basin_id):
        self.working_path = working_path
        self.basin_id = basin_id

    def check_validation(self, basin_list, basin_id):
        assert isinstance(basin_id, str)
        assert (len(basin_id) == 8 and basin_id.isdigit())
        assert (basin_id in basin_list.values)

    #D:\MyGIS\hydro_dl_project\camels_data\basin_mean_forcing\daymet\01
    def load_forcing_data(self, working_path, huc_id, basin_id):
        forcing_path = os.path.join(working_path, 'camels_data', 'basin_mean_forcing', 'daymet', huc_id,
                                    basin_id + '_lump_cida_forcing_leap.txt')
        forcing_data = pd.read_csv(forcing_path, sep="\s+|;|:", header=0, skiprows=3, engine='python')
        forcing_data.rename(columns={"Mnth": "Month"}, inplace=True)
        forcing_data['date'] = pd.to_datetime(forcing_data[['Year', 'Month', 'Day']])
        forcing_data['dayl(day)'] = forcing_data['dayl(s)'] / 86400
        forcing_data['tmean(C)'] = (forcing_data['tmin(C)'] + forcing_data['tmax(C)']) / 2


        with open(forcing_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])
        return forcing_data, area

    #D:\MyGIS\hydro_dl_project\camels_data\usgs_streamflow\01
    def load_flow_data(self, working_path, huc_id, basin_id, area):
        flow_path =os.path.join(working_path, 'camels_data', 'usgs_streamflow', huc_id,
                                basin_id + '_streamflow_qc.txt')
        flow_data = pd.read_csv(flow_path, sep="\s+", names=['Id', 'Year', 'Month', 'Day', 'Q', 'QC'],
                                header=None, engine='python')
        flow_data['date'] = pd.to_datetime(flow_data[['Year', 'Month', 'Day']])
        flow_data['flow(mm)'] = 28316846.592 * flow_data['Q'] * 86400 / (area * 10 ** 6)  # （英尺/淼转为毫米/秒） -> 转为毫米/天  ->
        return flow_data

    def load_data(self):
        basin_list = pd.read_csv(os.path.join(self.working_path, 'basin_list.txt'),
                                 sep='\t', header=0, dtype={'HUC': str, 'BASIN_ID': str})
        self.check_validation(basin_list, self.basin_id)
        huc_id = basin_list[basin_list['BASIN_ID'] == self.basin_id]['HUC'].values[0]
        #print(huc_id)
        forcing_data, area = self.load_forcing_data(self.working_path, huc_id, self.basin_id)
        flow_data = self.load_flow_data(self.working_path, huc_id, self.basin_id, area)
        merged_data = pd.merge(forcing_data, flow_data, on='date')
        merged_data = merged_data[(merged_data['date'] >= datetime(1980, 10, 1)) &
                                  (merged_data['date'] <= datetime(2010, 9, 30))]
        merged_data = merged_data.set_index('date')
        pd_data = merged_data[['prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)', 'flow(mm)']]
        return pd_data


