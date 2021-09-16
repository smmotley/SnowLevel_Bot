import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from datetime import datetime
#import panel as pn

class GRIB_DL():
    DEFAULT_ENDPOINT = 'nomads.ncep.noaa.gov/dods/'
    def __init__(self, model=None, model_resolution='', model_run=None, date=None):
        if not model:
            model = 'gfs'
        if not model_run:
            model_run = '06z'
        if not date:
            date = datetime.utcnow().strftime("%Y%m%d")
        url = f"{model}/{model}{date}/{model}_{model_resolution}_{model_run}"
        if model == 'gfs':
            url = f"{model}_{model_resolution}/{model}{date}/{model}_{model_resolution}_{model_run}"
        if 'nam' in model:
            url = f"{model}/{model}{date}/{model}_{model_resolution}_{model_run}"
        self.url = f"https://{self.DEFAULT_ENDPOINT}{url}"
        self.vtimes = self.valid_times()

    def valid_times(self):
        vtimes=[]
        ds = xr.open_dataset(self.url)
        for t in range(ds.time.size):
            vtimes.append(datetime.utcfromtimestamp(ds.time[t].data.astype('O') / 1e9))
        return vtimes

    def pull_point_data(self, lat, lon, variable, level, time=None):
        ds = xr.open_dataset(self.url)
        if time:
            if level == 'surface' or level == 'sfc':     # Remove level parameter or this will fail (no lev param)
                data = ds[variable].sel(time=time, lon=lon, lat=lat, method='nearest')
            else:
                data = ds[variable].sel(time=time, lon=lon, lat=lat, lev=level, method='nearest')
        else:
            if level == 'surface' or level == 'sfc':     # Remove level parameter or this will fail (no lev param)
                data = ds[variable].sel(lon=lon, lat=lat, method='nearest')
            else:
                data = ds[variable].sel(lon=lon, lat=lat, lev=level, method='nearest')
        return data

    def pull_global_data(self, variable, level=None, time=None):
        ds = xr.open_dataset(self.url)
        if time and level:
            data = ds[variable].sel(time=time, lev=level)
        elif level:
            data = ds[variable].sel(lev=level)
        else:
            data = ds[variable]
        return data


    def plot_ds(self, ds, time, bounds=None, show=False):
        ax = plt.axes(projection=ccrs.PlateCarree())
        if not bounds:
            bounds = [-130,-80,29,48] # West Lon, East Lon, South Lat, North Lat
        ax.set_extent(bounds)
        ax.add_feature(cfeature.STATES.with_scale('50m'))
        dsp = ds.sel(time=time)
        dsp = dsp.where(dsp.data > 0)
        dsp.plot()
        if show:
            plt.show()
        return

    def interactive_plot(self, ds, time, bounds=None, show=False):
        gv.extension('bokeh')
        dsp = ds.sel(time=time)
        dsp = dsp.where(dsp.data > 0)
        refl = gv.Dataset(dsp, ['lon', 'lat', 'time'], 'refd1000m', crs=ccrs.PlateCarree())
        images = refl.to(gv.Image)
        #regridded_img = regrid(images)
        images.opts(cmap='viridis', colorbar=True, width=600, height=500) * gv.tile_sources.ESRI.options(show_bounds=True)
        #pn.panel(images).show()
        gv.save(images,'image.html')
        return

class Parameter_Builder(GRIB_DL):
    def __init__(self, models, model_resolution='', model_runs=None, dates=None):
        if not models:
            models = ['gfs']
        if not model_runs:
            model_runs = ['06z']
        if not dates:
            dates = [datetime.utcnow().strftime("%Y%m%d")]
        for model in models:
            for date in dates:
                for model_run in model_runs:
                    url = f"{model}/{model}{date}/{model}_{model_resolution}_{model_run}"
                    if model == 'gfs':
                        url = f"{model}_{model_resolution}/{model}{date}/{model}_{model_resolution}_{model_run}"
                    if 'nam' in model:
                        url = f"{model}/{model}{date}/{model}_{model_resolution}_{model_run}"
                    self.url = f"https://{self.DEFAULT_ENDPOINT}{url}"
                    self.vtimes = self.valid_times()